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
    """
    문장 내/외부에 포함된 [AI가 분석한 강의 요약] 등 불필요한 메타 구문을
    강력하게 제거하는 헬퍼 함수
    """
    if not text:
        return ""

    # 1. 대괄호 및 안의 내용 전체 삭제 (예: [AI가 분석한 강의 요약], [요약] 등)
    cleaned = re.sub(r"\[.*?\]", "", text)

    # 2. 대괄호가 없는 형태의 AI 요약/헤더 메타 문구 제거
    cleaned = re.sub(
        r"(AI가\s*분석한\s*강의\s*요약|강의\s*요약|요약\s*본문|학습\s*목표|목차|참고자료)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 3. 마크다운 특수문자(#, *, `, _, >, ~ 등) 및 앞쪽 숫자/기호 목록 제거
    cleaned = re.sub(r"[#\*`_>~]", " ", cleaned)
    cleaned = re.sub(r"^[0-9\.\-\s\:\;]+", "", cleaned).strip()

    return cleaned


def filter_valid_sentences(summary_text):
    """
    강의 요약 본문에서 메타문구를 완벽히 제거하고
    순수한 '단일 명제 문장'만 추출합니다.
    """
    if not summary_text:
        return []

    # 1차 전처리: 전체 텍스트에서 대괄호 및 메타구문 1차 삭제
    text_clean = clean_quiz_text(summary_text)

    # KiWi 문장 분리
    try:
        raw_sents = [s.text.strip() for s in kiwi.split_into_sents(text_clean)]
    except Exception:
        raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_clean)]

    valid_sentences = []

    # O/X 명제로 사용할 수 없는 금지 키워드
    EXCLUDE_KEYWORDS = [
        "분석한", "요약", "다음은", "소개", "목차", "학습 목표",
        "무엇인가", "알아봅시다", "설명하시오", "출처", "작성일",
        "첫째", "둘째", "셋째", "결론적으로", "위의 내용"
    ]

    for sent in raw_sents:
        # 2차 전처리: 각 문장별로 혹시 남아있을 지 모르는 메타 텍스트 완벽 지우기
        sent_str = clean_quiz_text(sent)

        # 1) 길이기준: 너무 짧거나(단순 헤더) 너무 긴 문장 스킵
        if len(sent_str) < 15 or len(sent_str) > 100:
            continue

        # 2) 금지 키워드 검사
        if any(kw in sent_str for kw in EXCLUDE_KEYWORDS):
            continue

        # 3) 질문 형태(~입니까?, ?, 무엇 등) 제외
        if sent_str.endswith("?") or "무엇" in sent_str or "어떻게" in sent_str:
            continue

        # 4) KiWi 형태소 분석으로 완전한 서술문 구조(명사 + 동사/형용사) 검증
        tokens = kiwi.tokenize(sent_str)
        pos_tags = [t.tag for t in tokens]

        has_noun = any(tag.startswith("N") for tag in pos_tags)
        has_verb_or_adj = any(tag.startswith("V") or tag.startswith("X") for tag in pos_tags)
        ends_with_ef = any(tokens[-1].tag.startswith(t) for t in ["EF", "SF"]) or sent_str.endswith(("다.", "다"))

        if has_noun and has_verb_or_adj and ends_with_ef:
            valid_sentences.append(sent_str)

    # 중복 문장 제거
    return list(dict.fromkeys(valid_sentences))


def generate_tf_quiz_items(summary_text, question_count=20):
    """
    정제된 문장으로 O/X 퀴즈 항목을 생성합니다.
    """
    sentences = filter_valid_sentences(summary_text)

    if not sentences:
        return []

    all_nouns = extract_keywords_with_kiwi_for_objective(summary_text)
    selected_sentences = sentences[:question_count]
    quiz_items = []

    model = get_sbert_model()

    for idx, sent in enumerate(selected_sentences, start=1):
        # 3차 전처리: final 퀴즈 text 직전 단 한번 더 검증
        sent_final = clean_quiz_text(sent)
        is_true = (idx % 2 != 0)  # 홀수 번호: O, 짝수 번호: X

        if is_true:
            quiz_items.append({
                "number": idx,
                "question_text": sent_final,
                "correct_answer": "O",
                "explanation": "강의 본문 내용과 정확히 일치하는 명제입니다.",
                "original_sentence": sent_final,
            })
        else:
            sent_nouns = extract_keywords_with_kiwi_for_objective(sent_final)
            distorted_sent = sent_final
            replaced_word = ""
            new_word = ""

            if sent_nouns and len(all_nouns) > 3:
                # 문장 안에 실제로 존재하는 명제 단어 선택
                target_word = None
                for n in sent_nouns:
                    if n in sent_final and len(n) > 1:
                        target_word = n
                        break

                if target_word:
                    candidates = [n for n in all_nouns if n != target_word and len(n) > 1]

                    if candidates:
                        try:
                            target_emb = model.encode(target_word, convert_to_tensor=True)
                            cand_embs = model.encode(candidates, convert_to_tensor=True)
                            scores = util.cos_sim(target_emb, cand_embs)

                            # 유사도 텐서 치수 안전 처리
                            if scores.dim() > 1:
                                scores = scores.squeeze(0)

                            top_idx = int(torch.argmax(scores))
                            candidate_word = candidates[top_idx]

                            # 실제 치환이 성공한 경우에만 변수 업데이트
                            if target_word in sent_final:
                                distorted_sent = sent_final.replace(target_word, candidate_word, 1)
                                replaced_word = target_word
                                new_word = candidate_word
                        except Exception:
                            # SBERT 처리 실패 시 기본 폴백
                            pass

            explanation = (
                f"강의 본문의 '{replaced_word}'(이)가 '{new_word}'(으)로 잘못 변경된 명제입니다."
                if (replaced_word and new_word) else "강의 내용과 일치하지 않는 명제입니다."
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
    """
    생성된 O/X 문항들을 QuizQuestion 모델에 저장합니다.
    """
    # 1. 부모 Quiz 모델의 explanation 필드가 "tf"로 설정되도록 보장
    if quiz.explanation != TF_QUIZ_TYPE:
        quiz.explanation = TF_QUIZ_TYPE
        quiz.save(update_fields=["explanation"])

    # 2. 기존 문제 삭제 후 재생성
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
