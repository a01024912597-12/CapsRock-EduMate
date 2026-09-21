import os
import re
import json
import threading

import torch
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
from django.conf import settings

from ..models import Quiz, QuizQuestion, QuizAnswer
from .objective_quiz_service import extract_keywords_with_kiwi_for_objective

TF_QUIZ_TYPE = "tf"
SBERT_MODEL_NAME = getattr(
    settings,
    "OBJECTIVE_SBERT_MODEL_NAME",
    os.environ.get("OBJECTIVE_SBERT_MODEL_NAME", "jhgan/ko-sroberta-multitask"),
)

_sbert_model = None
_sbert_lock = threading.Lock()
kiwi = Kiwi()


def get_sbert_model():
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model

    with _sbert_lock:
        if _sbert_model is None:
            _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    return _sbert_model


def clean_quiz_text(text):
    if not text:
        return ""

    cleaned = re.sub(r"\[.*?\]", "", text)
    cleaned = re.sub(
        r"(AI가\s*분석한\s*강의\s*요약|강의\s*요약|요약\s*본문|학습\s*목표|목차|참고자료)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[#\*`_>~]", " ", cleaned)

    # [수정] '숫자 + 세기' 및 순번 표기 보호를 위해 앞쪽 마크다운식 목록 숫자만 선택 제거
    cleaned = re.sub(r"^[0-9]+[\.\)\-]\s*", "", cleaned).strip()

    return cleaned


def extract_nouns_from_sent(sentence):
    """
    단일 문장에서 명사를 추출하되,
    '숫자 + 세기' 패턴(예: 20세기, 21세기, 19세기)은 하나의 단위로 결합하여 추출합니다.
    """
    tokens = kiwi.tokenize(sentence)
    raw_nouns = [t.form for t in tokens if t.tag.startswith("N") and len(t.form) > 1]

    # '숫자+세기' (예: 21세기, 20세기) 패턴 감지 및 수집
    century_matches = re.findall(r"\b\d+\s*세기\b", sentence)

    final_nouns = []
    for noun in raw_nouns:
        # 단독으로 '세기'만 추출된 경우 제거 (숫자와 결합된 표기로만 사용하도록)
        if noun == "세기":
            continue
        final_nouns.append(noun)

    # '숫자 세기'를 키워드 목록에 통합
    for c_match in century_matches:
        c_clean = c_match.replace(" ", "")
        if c_clean not in final_nouns:
            final_nouns.append(c_clean)

    return final_nouns


def filter_valid_sentences(summary_text):
    valid_sentences = []

    if not summary_text:
        return valid_sentences

    text_clean = clean_quiz_text(summary_text)

    try:
        raw_sents = [s.text.strip() for s in kiwi.split_into_sents(text_clean)]
    except Exception:
        raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_clean)]

    EXCLUDE_KEYWORDS = [
        "분석한", "요약", "다음은", "소개", "목차", "학습 목표",
        "무엇인가", "알아봅시다", "설명하시오", "출처", "작성일",
        "첫째", "둘째", "셋째", "결론적으로", "위의 내용"
    ]

    processed_sents = []
    for s in raw_sents:
        s_clean = clean_quiz_text(s)
        if len(s_clean) < 15 or any(kw in s_clean for kw in EXCLUDE_KEYWORDS):
            continue
        if s_clean.endswith("?") or "무엇" in s_clean or "어떻게" in s_clean:
            continue

        nouns = extract_nouns_from_sent(s_clean)
        if nouns:
            processed_sents.append({"text": s_clean, "nouns": set(nouns)})

    # 키워드 공유 기반 문장 결합
    i = 0
    while i < len(processed_sents):
        curr = processed_sents[i]
        group = [curr["text"]]
        group_nouns = set(curr["nouns"])

        j = i + 1
        while j < len(processed_sents) and len(group) < 3:
            next_sent = processed_sents[j]
            common_keywords = group_nouns.intersection(next_sent["nouns"])
            if common_keywords or len(group) == 1:
                group.append(next_sent["text"])
                group_nouns.update(next_sent["nouns"])
                j += 1
            else:
                break

        combined_text = " ".join(group).strip()

        if 70 <= len(combined_text) <= 280:
            valid_sentences.append(combined_text)

        i = j if j > i + 1 else i + 1

    return list(dict.fromkeys(valid_sentences))


def generate_tf_quiz_items(summary_text, question_count=20):
    sentences = filter_valid_sentences(summary_text)

    if not sentences:
        return []

    # 전체 본문 키워드 추출 시에도 '숫자+세기' 보존
    raw_all_nouns = extract_keywords_with_kiwi_for_objective(summary_text)
    century_matches_all = re.findall(r"\b\d+\s*세기\b", summary_text)

    all_nouns = [n for n in raw_all_nouns if n != "세기"]
    for cm in century_matches_all:
        cm_clean = cm.replace(" ", "")
        if cm_clean not in all_nouns:
            all_nouns.append(cm_clean)

    selected_sentences = sentences[:question_count]
    quiz_items = []

    model = get_sbert_model()

    for idx, sent in enumerate(selected_sentences, start=1):
        sent_final = clean_quiz_text(sent)
        is_true = (idx % 2 != 0)

        if is_true:
            quiz_items.append({
                "number": idx,
                "question_text": sent_final,
                "correct_answer": "O",
                "explanation": "강의 본문의 키워드 설명 내용 및 맥락과 정확히 일치하는 명제입니다.",
                "original_sentence": sent_final,
            })
        else:
            sent_nouns = extract_nouns_from_sent(sent_final)
            distorted_sent = sent_final
            replaced_word = ""
            new_word = ""

            if sent_nouns and len(all_nouns) > 3:
                # 타겟 키워드 선정 ('세기' 단독 단어 제외)
                target_word = next((n for n in sent_nouns if n in sent_final and n != "세기" and len(n) > 1), None)

                if target_word:
                    candidates = [n for n in all_nouns if n != target_word and n != "세기" and len(n) > 1]

                    if candidates:
                        try:
                            target_emb = model.encode(target_word, convert_to_tensor=True)
                            cand_embs = model.encode(candidates, convert_to_tensor=True)
                            scores = util.cos_sim(target_emb, cand_embs)

                            if scores.dim() > 1:
                                scores = scores.squeeze(0)

                            top_idx = int(torch.argmax(scores))
                            candidate_word = candidates[top_idx]

                            if target_word in sent_final:
                                distorted_sent = sent_final.replace(target_word, candidate_word, 1)
                                replaced_word = target_word
                                new_word = candidate_word
                        except Exception:
                            pass

            explanation = (
                f"강의 본문의 핵심 개념인 '{replaced_word}'(이)가 '{new_word}'(으)로 잘못 설명된 명제입니다."
                if (replaced_word and new_word) else "강의 내용의 맥락과 일치하지 않는 명제입니다."
            )

            quiz_items.append({
                "number": idx,
                "question_text": distorted_sent,
                "correct_answer": "X",
                "explanation": explanation,
                "original_sentence": sent_final,
            })

    return quiz_items


def save_tf_questions(quiz, quiz_items):
    if quiz.explanation != TF_QUIZ_TYPE:
        quiz.explanation = TF_QUIZ_TYPE
        quiz.save(update_fields=["explanation"])

    QuizQuestion.objects.filter(quiz=quiz).delete()

    for item in quiz_items:
        meta = {
            "explanation": item["explanation"],
            "original_sentence": item["original_sentence"],
        }
        QuizQuestion.objects.create(
            quiz=quiz,
            number=item["number"],
            question_text=item["question_text"],
            model_answer=item["correct_answer"],
            explanation=json.dumps(meta, ensure_ascii=False),
        )