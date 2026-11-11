import os
import re
import math
import threading

import torch
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from django.conf import settings

from ..utils.text_utils import normalize_text_for_keyword
from ..utils.quiz_language import LANG_KO, LANG_EN, LANG_ZH, detect_quiz_language

# Reranker 기반 채점 모델 설정
RERANKER_MODEL_NAME = getattr(
    settings,
    "RERANKER_MODEL_NAME",
    os.environ.get("RERANKER_MODEL_NAME", "dragonkue/bge-reranker-v2-m3-ko"),
)
RERANKER_MAX_LENGTH = 512
_reranker_tokenizer = None
_reranker_model = None
_reranker_lock = threading.Lock()

kiwi = Kiwi()



def _grading_language(*texts):
    return detect_quiz_language("\n".join(str(text or "") for text in texts))


def _grading_text(language, key, **kwargs):
    table = {
        LANG_KO: {
            "required_missing": "문제의 핵심 개념 '{term}'이(가) 답안에 명시되지 않았습니다.",
            "empty": "답안이 작성되지 않았거나 성의가 부족합니다. 모범 답안을 확인하고 다시 작성해보세요.",
            "reference": "💡 **모범 답안 참조**: {model_answer}",
            "exact": "모범 답안과 정확히 일치하는 완벽한 답안입니다!",
            "exact_detail": "모범 답안의 모든 요소를 완벽하게 포함하여 감점 요소가 없습니다.",
            "unavailable": "자동 채점 모델을 사용할 수 없어 점수 판정을 보류했습니다. 답안은 저장되었으며 검토가 필요합니다.",
            "short": "답안이 너무 짧아 모범 답안의 핵심 개념을 충분히 설명하지 못했습니다.",
            "missing_some": "모범 답안의 필수 키워드({keywords} 등)가 누락되어 설명의 완성도가 떨어집니다.",
            "missing_many": "모범 답안의 핵심 키워드가 다수 누락되었습니다.",
            "good_keywords": "모범 답안의 핵심 개념과 키워드를 대부분 잘 활용하여 작성했습니다.",
            "covered": "✅ **잘 서술된 부분**:\n모범 답안의 주요 내용인 '{text}' 관련 개념을 잘 언급했습니다.",
            "uncovered": "⚠️ **보완이 필요한 부분 (모범 답안 대비)**:\n모범 답안의 핵심 설명인 '{text}' 내용이 다소 부족합니다.",
            "missing_terms": "🔑 **추가해야 할 핵심 용어**: {keywords}",
            "query": "질문: {question} 정답: {answer}",
        },
        LANG_EN: {
            "required_missing": "The key concept '{term}' from the question is not stated in the answer.",
            "empty": "The answer is empty or too limited to grade reliably. Review the model answer and try again.",
            "reference": "💡 **Model answer reference**: {model_answer}",
            "exact": "The answer exactly matches the model answer.",
            "exact_detail": "The answer contains all elements of the model answer with no deduction needed.",
            "unavailable": "The automatic grading model is unavailable, so the score was withheld. The answer was saved for review.",
            "short": "The answer is too short to explain the key concept sufficiently.",
            "missing_some": "Important terms from the model answer ({keywords}, etc.) are missing, so the explanation is incomplete.",
            "missing_many": "Several key terms from the model answer are missing.",
            "good_keywords": "The answer uses most of the key concepts and terms from the model answer well.",
            "covered": "✅ **Well explained**:\nThe answer covers the model-answer idea '{text}'.",
            "uncovered": "⚠️ **Needs improvement**:\nThe model-answer point '{text}' is not sufficiently explained.",
            "missing_terms": "🔑 **Key terms to add**: {keywords}",
            "query": "Question: {question} Answer: {answer}",
        },
        LANG_ZH: {
            "required_missing": "答案中没有明确写出问题的核心概念“{term}”。",
            "empty": "答案为空或内容过少，无法可靠评分。请参考标准答案后重新作答。",
            "reference": "💡 **标准答案参考**：{model_answer}",
            "exact": "答案与标准答案完全一致。",
            "exact_detail": "答案完整包含标准答案的全部要点，无需扣分。",
            "unavailable": "自动评分模型当前不可用，因此暂不判定分数。答案已保存，可供后续复核。",
            "short": "答案过短，未能充分说明核心概念。",
            "missing_some": "标准答案中的重要关键词（{keywords}等）缺失，说明不够完整。",
            "missing_many": "标准答案中的多个核心关键词缺失。",
            "good_keywords": "答案较好地使用了标准答案中的大部分核心概念和关键词。",
            "covered": "✅ **说明较好的部分**：\n答案涉及了标准答案中的“{text}”。",
            "uncovered": "⚠️ **需要补充的部分**：\n标准答案中的“{text}”说明得还不充分。",
            "missing_terms": "🔑 **建议补充的核心词**：{keywords}",
            "query": "问题：{question} 答案：{answer}",
        },
    }
    return table.get(language, table[LANG_KO]).get(key, "").format(**kwargs)


def _extract_grading_keywords_non_ko(text, language):
    text = text or ""
    if language == LANG_EN:
        stop = {
            "the", "and", "for", "that", "with", "from", "this", "into", "than", "then", "which",
            "what", "why", "how", "are", "was", "were", "is", "be", "been", "being", "have", "has",
            "had", "can", "could", "should", "would", "may", "might", "must", "answer", "question",
            "explain", "describe", "meaning", "feature", "important", "lecture", "model",
        }
        values = re.findall(r"\b[A-Za-z][A-Za-z0-9.+#'/-]{2,}\b", text)
        result = []
        for value in values:
            if value.casefold() in stop:
                continue
            if value.casefold() not in {item.casefold() for item in result}:
                result.append(value)
        return result[:20]

    stop = {"答案", "问题", "课程", "概念", "含义", "特点", "重要", "说明", "核心", "标准答案"}
    values = re.findall(r"[\u4e00-\u9fff]{2,8}|\b[A-Za-z][A-Za-z0-9.+#'/-]{1,}\b", text)
    result = []
    for value in values:
        if value in stop:
            continue
        if value not in result:
            result.append(value)
    return result[:20]


def get_reranker_model():
    """답안 채점용 Reranker 모델을 지연 로딩한다.

    서버 시작 시 바로 로딩하지 않고,
    사용자가 답안을 저장하는 순간 처음 한 번만 로딩한다.
    """
    global _reranker_tokenizer, _reranker_model

    if _reranker_tokenizer is not None and _reranker_model is not None:
        return _reranker_tokenizer, _reranker_model

    with _reranker_lock:
        if _reranker_tokenizer is None or _reranker_model is None:
            print(f"[Reranker 로딩] 모델 로딩 시작: {RERANKER_MODEL_NAME}")

            _reranker_tokenizer = AutoTokenizer.from_pretrained(
                RERANKER_MODEL_NAME,
                trust_remote_code=True,
            )
            _reranker_model = AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL_NAME,
                trust_remote_code=True,
            )
            _reranker_model.eval()

            print(f"[Reranker 로딩] 모델 로딩 완료: {RERANKER_MODEL_NAME}")

    return _reranker_tokenizer, _reranker_model


def sigmoid(value):
    """logit 값을 0~1 범위로 변환한다."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if value >= 60:
        return 1.0

    if value <= -60:
        return 0.0

    return 1 / (1 + math.exp(-value))


KEYWORD_STOPWORDS = {
    "사이", "평화", "자급", "자족", "농촌", "경제", "구조", "제품", "수출", "부진",
    "한계", "개항", "기대", "상품", "현상", "수요", "증가", "점령", "파괴", "체결",
    "피신", "대신", "중재", "마무리", "확대", "발생", "진행", "시작", "종결", "개시",
    "추진", "요구", "거부", "승인", "체포", "파견", "설립", "조약", "연합군", "전쟁",
    "배경", "원인", "결과", "의미", "특징", "영향", "과정", "역할", "상황", "사건",
    "내용", "문제", "답안", "설명", "개념", "강의", "관련", "해당", "부문", "측면",
    "입장", "경우", "주요", "핵심", "시기", "당시", "모습", "형태", "관계", "주체",
    "대상", "반발", "참전", "가세", "선포", "수색", "조치", "대응", "확산", "변화"
}


def extract_grading_keywords(text):
    """채점 보조용 핵심 키워드를 추출한다. 한국어는 기존 Kiwi 로직을 그대로 사용한다."""
    text = (text or "").strip()

    if not text:
        return []

    language = _grading_language(text)
    if language != LANG_KO:
        return _extract_grading_keywords_non_ko(text, language)

    keywords = []

    try:
        tokens = kiwi.tokenize(text)
        for token in tokens:
            word = token.form.strip()
            tag = token.tag

            # 체언(명사: NNG, NNP) 및 영문/외국어(SL) 키워드 대상
            if not (tag.startswith("N") or tag == "SL"):
                continue

            # 1. 2글자 미만 제외
            # 2. 강화된 KEYWORD_STOPWORDS에 포함된 단어 제외
            # 3. 글자가 너무 길거나(6글자 이상 일반 명사 등) 모호한 단어 필터링 강화
            if len(word) < 2 or word in KEYWORD_STOPWORDS:
                continue

            if word not in keywords:
                keywords.append(word)

    except Exception as e:
        print("[키워드 추출 에러]", e)
        return []

    return keywords


def calculate_keyword_coverage(reference_text, user_answer):
    """기준 텍스트의 핵심 키워드가 사용자 답안에 얼마나 포함되었는지 계산한다."""
    reference_keywords = extract_grading_keywords(reference_text)
    normalized_user_answer = normalize_text_for_keyword(user_answer)

    if not reference_keywords or not normalized_user_answer:
        return 0.0, [], reference_keywords

    matched_keywords = []
    missing_keywords = []

    for keyword in reference_keywords:
        normalized_keyword = normalize_text_for_keyword(keyword)
        if normalized_keyword and normalized_keyword in normalized_user_answer:
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    coverage = len(matched_keywords) / len(reference_keywords) if reference_keywords else 1.0
    return round(float(coverage), 4), matched_keywords, missing_keywords


def is_meaningless_answer(user_answer):
    """채점할 의미가 없는 답안인지 확인한다."""
    user_answer = (user_answer or "").strip()

    if not user_answer:
        return True

    compact = re.sub(r"\s+", "", user_answer)

    meaningless_values = {
        "모르겠다", "모름", "몰라", "없음", "없다",
        "잘모르겠다", "잘모름", "패스", "pass",
        "ㅇㅇ", "ㄴㄴ", "ㅋㅋ", "ㅎㅎ",
        "test", "테스트", "i don't know", "dont know", "unknown", "skip", "不知道", "不清楚", "跳过",
    }

    if compact.lower() in meaningless_values:
        return True

    if re.fullmatch(r"[\d\W_]+", compact):
        return True

    meaningful_chars = re.findall(r"[가-힣A-Za-z0-9]", compact)
    if len(meaningful_chars) < 2:
        return True

    return False


def extract_subject_entities(text):
    """Kiwi를 사용해 문장에서 주어/주체 역할을 하는 명사를 동적 추출한다."""
    entities = []
    try:
        tokens = kiwi.tokenize(text)
        for i, token in enumerate(tokens):
            if token.tag in ("NNP", "NNG") and len(token.form) >= 2:
                if i + 1 < len(tokens) and tokens[i + 1].tag in ("JKS", "JXC"):
                    if token.form not in entities:
                        entities.append(token.form)
    except Exception:
        pass
    return entities


def detect_factual_conflicts_generic(question_text, model_answer, user_answer):
    """하드코딩 없이 문맥 기반 사실관계 및 누락 검사."""
    conflicts = []

    question_keywords = extract_grading_keywords(question_text)
    strict_terms = [kw for kw in question_keywords if len(kw) >= 3]
    norm_user = re.sub(r"\s+", "", user_answer).lower()

    for term in strict_terms:
        norm_term = re.sub(r"\s+", "", term).lower()
        if norm_term not in norm_user:
            conflicts.append({
                "type": "required_term_missing",
                "message": _grading_text(_grading_language(question_text, model_answer, user_answer), "required_missing", term=term),
                "cap": 0.70,
            })

    return conflicts


def calculate_answer_length_score(model_answer, user_answer):
    """답안 길이의 적절성을 0~1 점수로 계산한다."""
    model_answer = (model_answer or "").strip()
    user_answer = (user_answer or "").strip()

    if is_meaningless_answer(user_answer):
        return 0.0

    model_len = len(model_answer)
    user_len = len(user_answer)

    if user_len == 0:
        return 0.0

    if user_len < 10:
        return 0.10

    if user_len < 20:
        return 0.35

    if not model_len:
        return 0.5

    ratio = user_len / model_len

    if ratio < 0.25:
        return 0.35

    if ratio < 0.40:
        return 0.60

    if ratio <= 1.60:
        return 1.0

    if ratio <= 2.20:
        return 0.80

    return 0.60


def calculate_reranker_score(question_text, model_answer, user_answer):
    """Reranker 모델로 문제/모범 답안과 사용자 답안의 관련성 점수를 계산한다."""
    question_text = (question_text or "").strip()
    model_answer = (model_answer or "").strip()
    user_answer = (user_answer or "").strip()

    if not model_answer or not user_answer:
        return 0.0

    language = _grading_language(question_text, model_answer)
    query_reference = _grading_text(language, "query", question=question_text, answer=model_answer)

    try:
        tokenizer, reranker = get_reranker_model()

        inputs = tokenizer(
            query_reference,
            user_answer,
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LENGTH,
            return_tensors="pt",
        )

        device = next(reranker.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = reranker(**inputs)
            logits = outputs.logits

        if logits.dim() == 2 and logits.size(-1) >= 2:
            probabilities = torch.softmax(logits[0], dim=-1)
            score = probabilities[-1].item()
        else:
            score = sigmoid(logits.view(-1)[0].item())

        return round(max(0.0, min(float(score), 1.0)), 4)

    except Exception as e:
        print("[Reranker 채점 에러]", e)
        return None


def generate_detailed_analysis(model_answer, user_answer, matched_kw, missing_kw, conflicts):
    """모범 답안과 학생 답안을 비교하여 요약 언어에 맞는 상세 분석 문구를 생성한다."""
    language = _grading_language(model_answer, user_answer)
    analysis_parts = []
    model_sentences = [s.strip() for s in re.split(r'(?<=[.?!。])\s+|\n+', model_answer) if s.strip()]
    covered_sentences = []
    uncovered_sentences = []

    for sent in model_sentences:
        sent_kw = [kw for kw in matched_kw if kw.casefold() in sent.casefold()]
        if sent_kw or any(kw.casefold() in user_answer.casefold() for kw in matched_kw if kw.casefold() in sent.casefold()):
            covered_sentences.append(sent)
        else:
            uncovered_sentences.append(sent)

    if covered_sentences:
        analysis_parts.append(_grading_text(language, "covered", text=" / ".join(covered_sentences[:2])))
    if uncovered_sentences:
        analysis_parts.append(_grading_text(language, "uncovered", text=" / ".join(uncovered_sentences[:2])))
    if missing_kw:
        analysis_parts.append(_grading_text(language, "missing_terms", keywords=", ".join(missing_kw)))
    return "\n\n".join(analysis_parts)


def calculate_grading_score(question_text, model_answer, user_answer):
    """Reranker + 키워드 + 답안 길이 + 사실 오류 감지를 반영한 모범 답안 밀착형 자동 채점 함수."""
    question_text = (question_text or "").strip()
    model_answer = (model_answer or "").strip()
    user_answer = (user_answer or "").strip()

    language = _grading_language(question_text, model_answer, user_answer)

    feedback = {
        "score": 0.0,
        "language": language,
        "matched_keywords": [],
        "missing_keywords": [],
        "messages": [],
        "detail_analysis": "",
    }

    if not user_answer or is_meaningless_answer(user_answer):
        feedback["messages"].append(_grading_text(language, "empty"))
        feedback["detail_analysis"] = _grading_text(language, "reference", model_answer=model_answer)
        return 0.0, feedback

    norm_model = re.sub(r"\s+", "", model_answer).lower()
    norm_user = re.sub(r"\s+", "", user_answer).lower()
    if norm_model and norm_model == norm_user:
        feedback["score"] = 1.0
        feedback["messages"].append(_grading_text(language, "exact"))
        feedback["detail_analysis"] = _grading_text(language, "exact_detail")
        return 1.0, feedback

    reranker_score = calculate_reranker_score(question_text, model_answer, user_answer) if model_answer else None

    keyword_score, matched_kw, missing_kw = calculate_keyword_coverage(model_answer, user_answer)

    feedback["matched_keywords"] = matched_kw
    feedback["missing_keywords"] = missing_kw

    length_score = calculate_answer_length_score(model_answer, user_answer)

    if reranker_score is None:
        feedback["score"] = None
        feedback["grading_status"] = "review_required"
        feedback["messages"].append(_grading_text(language, "unavailable"))
        return None, feedback
    else:
        final_score = (reranker_score * 0.55) + (keyword_score * 0.35) + (
                length_score * 0.10)

    user_len = len(user_answer)
    if user_len < 15:
        final_score = min(final_score, 0.30)
        feedback["messages"].append(_grading_text(language, "short"))

    if keyword_score < 0.30:
        final_score = min(final_score, 0.60)
        if missing_kw:
            feedback["messages"].append(_grading_text(language, "missing_some", keywords=", ".join(missing_kw[:3])))
        else:
            feedback["messages"].append(_grading_text(language, "missing_many"))
    elif keyword_score >= 0.80:
        feedback["messages"].append(_grading_text(language, "good_keywords"))

    # 질문 용어의 반복 여부는 사실 오류나 필수 답안 요소의 근거가 아니다.
    conflicts = []
    if conflicts:
        min_cap = min(c.get("cap", 1.0) for c in conflicts)
        final_score = min(final_score, min_cap)
        for c in conflicts:
            feedback["messages"].append(f"🚨 {c['message']}")

    feedback["detail_analysis"] = generate_detailed_analysis(
        model_answer=model_answer,
        user_answer=user_answer,
        matched_kw=matched_kw,
        missing_kw=missing_kw,
        conflicts=conflicts
    )

    final_score = round(max(0.0, min(final_score, 1.0)), 4)
    feedback["score"] = final_score

    return final_score, feedback


def predict_answer_label(grading_score):
    """자동 채점 점수를 기반으로 평가 라벨을 반환한다."""
    if grading_score is None:
        return "검토 필요"
    if grading_score >= 0.90:
        return "정답"

    if grading_score >= 0.60:
        return "부분 정답"

    return "오답"
