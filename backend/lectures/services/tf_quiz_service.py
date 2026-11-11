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
from ..utils.quiz_language import (
    LANG_KO,
    LANG_EN,
    LANG_ZH,
    detect_quiz_language,
    split_sentences_multilingual,
    extract_non_korean_keywords,
    localized_quiz_text,
)

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


def _generate_tf_quiz_items_ko(summary_text, question_count=20):
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



def _clean_tf_text_multilingual(text):
    """영어/중국어 O/X 생성 전용 전처리. 섹션 라벨·이미지·HTML을 제거한다."""
    if not text:
        return ""

    language = detect_quiz_language(text)
    cleaned = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<script\b[^>]*>.*?</script>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]\((?:https?://|/)[^)]+\)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"▶\s*\d{1,3}:\d{2}[^\n]*", " ", cleaned)
    cleaned = re.sub(r"(AI\s*Lecture\s*Summary|Lecture\s*Summary|课程摘要|AI课程摘要)", "", cleaned, flags=re.IGNORECASE)

    en_meta = {
        "type", "basis", "meaning", "context", "exam point", "exam points",
        "short answer", "comparison", "sequence", "essay", "one-line final summary",
        "one line final summary",
    }
    zh_meta = {"类型", "依据", "含义", "语境", "考点", "简答", "比较", "顺序", "论述", "一句话总结"}

    kept_lines = []
    for raw_line in cleaned.splitlines():
        line = re.sub(r"^[\s#>*\-•]+", "", raw_line).strip()
        line = re.sub(r"[*_`~]+", "", line).strip()
        if not line:
            continue
        if re.search(r"(?:font-size|timeline-image-wrapper|https?://|\.jpg|\.png)", line, re.IGNORECASE):
            continue
        head = re.split(r"[:：]", line, maxsplit=1)[0].strip().casefold()
        if language == LANG_EN and head in en_meta:
            continue
        if language == LANG_ZH and re.split(r"[:：]", line, maxsplit=1)[0].strip() in zh_meta:
            continue
        # 숫자만 있는 섹션 제목 또는 '1. ...' 식 제목은 제외한다.
        if re.fullmatch(r"\d+\.?\s*", line):
            continue
        kept_lines.append(line)

    return " ".join(kept_lines)


def _filter_valid_sentences_non_ko(summary_text, language):
    """영어/중국어 요약에서 O/X용 서술문만 추출한다."""
    text_clean = _clean_tf_text_multilingual(summary_text)
    raw_sentences = split_sentences_multilingual(text_clean, language)

    exclude_keywords = (
        (
            "lecture type",
            "overall flow",
            "detailed lecture",
            "essential concepts",
            "key exam points",
            "exam point",
        )
        if language == LANG_EN
        else (
            "课程类型",
            "整体流程",
            "详细讲义",
            "核心概念",
            "考试重点",
            "关键考点",
        )
    )

    valid = []
    for sent in raw_sentences:
        sent = re.sub(r"\s+", " ", sent).strip(" -•:")
        if not sent:
            continue
        if any(keyword.casefold() in sent.casefold() for keyword in exclude_keywords):
            continue
        if sent.endswith("?") or sent.endswith("？"):
            continue

        if language == LANG_EN:
            if len(sent) < 25 or len(sent) > 320:
                continue
            if len(re.findall(r"[A-Za-z]{2,}", sent)) < 5:
                continue
        else:
            if len(sent) < 12 or len(sent) > 180:
                continue
            if len(re.findall(r"[\u4e00-\u9fff]", sent)) < 6:
                continue

        valid.append(sent)

    return list(dict.fromkeys(valid))


def _generate_tf_quiz_items_non_ko(summary_text, question_count=20, language=LANG_EN):
    """영어/중국어 요약 전용 O/X 생성기. 한국어 기존 로직과 분리한다."""
    sentences = _filter_valid_sentences_non_ko(summary_text, language)
    if not sentences:
        return []

    concept_keywords = []
    try:
        from .summary_service import extract_concept_entries_from_summary
        concept_keywords = [
            (entry.get("keyword") or "").strip()
            for entry in extract_concept_entries_from_summary(summary_text)
            if (entry.get("keyword") or "").strip()
        ]
    except Exception:
        concept_keywords = []

    all_keywords = extract_non_korean_keywords(
        summary_text,
        language,
        concept_keywords=concept_keywords,
    )

    quiz_items = []
    for sent in sentences:
        if len(quiz_items) >= question_count:
            break

        item_number = len(quiz_items) + 1
        make_true = item_number % 2 != 0

        if make_true:
            quiz_items.append({
                "number": item_number,
                "question_text": sent,
                "correct_answer": "O",
                "explanation": localized_quiz_text(language, "tf_true"),
                "original_sentence": sent,
            })
            continue

        sentence_terms = [
            term for term in all_keywords
            if len(term) > 1 and term in sent
        ]
        target_word = sentence_terms[0] if sentence_terms else None
        candidates = [
            term for term in all_keywords
            if target_word
            and term.casefold() != target_word.casefold()
            and term not in sent
            and len(term) > 1
        ]

        if not target_word or not candidates:
            continue

        candidates.sort(
            key=lambda value: (abs(len(value) - len(target_word)), value.casefold())
        )
        new_word = candidates[0]
        distorted_sent = sent.replace(target_word, new_word, 1)

        if distorted_sent == sent:
            continue

        quiz_items.append({
            "number": item_number,
            "question_text": distorted_sent,
            "correct_answer": "X",
            "explanation": localized_quiz_text(
                language,
                "tf_false",
                old=target_word,
                new=new_word,
            ),
            "original_sentence": sent,
        })

    print(
        f"[OX 퀴즈 생성] language={language}, "
        f"sentences={len(sentences)}, keywords={len(all_keywords)}, "
        f"generated={len(quiz_items)}"
    )
    return quiz_items


def generate_tf_quiz_items(summary_text, question_count=20):
    """요약 언어에 따라 기존 한국어 생성기 또는 다국어 생성기를 선택한다."""
    language = detect_quiz_language(summary_text)

    if language == LANG_KO:
        return _generate_tf_quiz_items_ko(
            summary_text=summary_text,
            question_count=question_count,
        )

    return _generate_tf_quiz_items_non_ko(
        summary_text=summary_text,
        question_count=question_count,
        language=language,
    )


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
