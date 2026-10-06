import os
import re
import math
import threading

import torch
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from django.conf import settings

from ..utils.text_utils import normalize_text_for_keyword

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

def extract_grading_keywords(text):
    """채점 보조용 핵심 키워드를 추출한다."""
    text = (text or "").strip()

    if not text:
        return []

    stopwords = {
        "것", "수", "등", "및", "때", "위해", "통해", "대한", "관련",
        "중심", "내용", "문제", "답안", "설명", "부분", "정도",
        "원인", "결과", "의미", "과정", "역할", "영향", "배경",
        "시작", "강화", "확대", "변화", "발생", "형성", "이후",
        "이전", "경우", "자신", "해당", "주요", "핵심", "강의",
        "사용자", "모범", "서술", "기반", "가능성",
    }

    keywords = []

    try:
        tokens = kiwi.tokenize(text)

        for token in tokens:
            word = token.form.strip()
            tag = token.tag

            if not tag.startswith("N"):
                continue

            if len(word) < 2:
                continue

            if word in stopwords:
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
        return 0.0, []

    matched_keywords = []

    for keyword in reference_keywords:
        normalized_keyword = normalize_text_for_keyword(keyword)

        if normalized_keyword and normalized_keyword in normalized_user_answer:
            matched_keywords.append(keyword)

    coverage = len(matched_keywords) / len(reference_keywords)

    return round(float(coverage), 4), matched_keywords

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
        "test", "테스트",
    }

    if compact.lower() in meaningless_values:
        return True

    # 숫자, 특수문자만 있는 답안: 123, !!! 등
    if re.fullmatch(r"[\d\W_]+", compact):
        return True

    meaningful_chars = re.findall(r"[가-힣A-Za-z]", compact)

    if len(meaningful_chars) < 2:
        return True

    return False

def normalize_for_fact_check(text):
    """사실 오류 검사에 사용할 텍스트를 정규화한다."""
    text = (text or "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("１", "1").replace("２", "2")
    text = text.replace("일차", "1차").replace("이차", "2차")
    return text

def contains_any(text, keywords):
    """문자열에 후보 키워드 중 하나라도 포함되어 있는지 확인한다."""
    text = normalize_for_fact_check(text)

    for keyword in keywords:
        if normalize_for_fact_check(keyword) in text:
            return True

    return False

def extract_required_concepts(question_text, model_answer):
    """문제와 모범답안에서 반드시 확인해야 할 핵심 개념을 추출한다."""
    source = f"{question_text} {model_answer}"

    known_concepts = [
        "애로호 사건",
        "임칙서",
        "임칙서의 아편 단속",
        "프랑스 선교사 처형 사건",
        "난징조약",
        "아편전쟁",
        "제1차 아편전쟁",
        "제2차 아편전쟁",
        "영프 연합군",
        "청나라",
        "영국",
        "프랑스",
    ]

    required = []

    for concept in known_concepts:
        if contains_any(source, [concept]) and concept not in required:
            required.append(concept)

    return required

def detect_factual_conflicts(question_text, model_answer, user_answer):
    """사용자 답안의 핵심 사실 오류를 감지한다.

    반환 예시:
    [
        {"type": "war_order_conflict", "message": "제2차 아편전쟁을 제1차 아편전쟁으로 작성함", "cap": 0.65}
    ]
    """
    question_norm = normalize_for_fact_check(question_text)
    model_norm = normalize_for_fact_check(model_answer)
    user_norm = normalize_for_fact_check(user_answer)

    conflicts = []

    # 1. 차수 오류: 제1차 / 제2차
    model_has_first = "제1차아편전쟁" in model_norm or "1차아편전쟁" in model_norm
    model_has_second = "제2차아편전쟁" in model_norm or "2차아편전쟁" in model_norm

    user_has_first = "제1차아편전쟁" in user_norm or "1차아편전쟁" in user_norm
    user_has_second = "제2차아편전쟁" in user_norm or "2차아편전쟁" in user_norm

    if model_has_second and user_has_first and not user_has_second:
        conflicts.append({
            "type": "war_order_conflict",
            "message": "제2차 아편전쟁을 제1차 아편전쟁으로 작성함",
            "cap": 0.65,
        })

    if model_has_first and user_has_second and not user_has_first:
        conflicts.append({
            "type": "war_order_conflict",
            "message": "제1차 아편전쟁을 제2차 아편전쟁으로 작성함",
            "cap": 0.65,
        })

    # 2. 핵심 사건명 대체 오류
    required_concepts = extract_required_concepts(question_text, model_answer)

    if "애로호 사건" in required_concepts:
        user_has_arrow = "애로호" in user_norm
        user_has_lin = "임칙서" in user_norm

        if not user_has_arrow and user_has_lin:
            conflicts.append({
                "type": "event_replacement_conflict",
                "message": "애로호 사건을 임칙서 사건 또는 임칙서 관련 내용으로 대체함",
                "cap": 0.55,
            })

    if "임칙서" in required_concepts or "임칙서의 아편 단속" in required_concepts:
        user_has_lin = "임칙서" in user_norm

        if not user_has_lin and "애로호" in user_norm:
            conflicts.append({
                "type": "event_replacement_conflict",
                "message": "임칙서의 아편 단속을 애로호 사건으로 대체함",
                "cap": 0.55,
            })

    if "프랑스 선교사 처형 사건" in required_concepts:
        has_french_missionary = (
            "프랑스선교사" in user_norm
            or "프랑스신부" in user_norm
            or "선교사처형" in user_norm
            or "신부처형" in user_norm
        )

        if not has_french_missionary and ("애로호" in user_norm or "임칙서" in user_norm):
            conflicts.append({
                "type": "event_replacement_conflict",
                "message": "프랑스 선교사 처형 사건을 다른 사건으로 대체함",
                "cap": 0.55,
            })

    # 3. 주체/대상 반전 오류
    # 예: 모범답안: 청나라에 밀수되던 영국의 아편
    #     사용자: 영국에 밀수되던 청나라의 아편
    model_has_british_opium_to_qing = (
        "청나라에밀수되던영국의아편" in model_norm
        or ("청나라에" in model_norm and "영국의아편" in model_norm)
    )
    user_has_qing_opium_to_britain = (
        "영국에밀수되던청나라의아편" in user_norm
        or ("영국에" in user_norm and "청나라의아편" in user_norm)
    )

    if model_has_british_opium_to_qing and user_has_qing_opium_to_britain:
        conflicts.append({
            "type": "subject_object_reversal",
            "message": "아편 밀수의 주체와 대상이 반대로 작성됨",
            "cap": 0.55,
        })


    # 3-1. 애로호 사건의 수색 주체/대상 반전 오류
    # 모범답안: 청나라 수군/관리 → 영국의 애로호 수색
    # 사용자답안: 영국 수군/관리 → 청나라의 애로호 수색
    model_has_qing_searched_british_arrow = (
        ("청나라수군" in model_norm or "청나라관리" in model_norm or "청나라가" in model_norm)
        and ("영국의애로호" in model_norm or "영국선박" in model_norm or "영국의선박" in model_norm)
        and ("수색" in model_norm or "체포" in model_norm)
    )

    user_has_britain_searched_qing_arrow = (
        ("영국수군" in user_norm or "영국관리" in user_norm or "영국이" in user_norm)
        and ("청나라의애로호" in user_norm or "청나라선박" in user_norm or "청나라의선박" in user_norm)
        and ("수색" in user_norm or "체포" in user_norm)
    )

    if model_has_qing_searched_british_arrow and user_has_britain_searched_qing_arrow:
        conflicts.append({
            "type": "arrow_incident_actor_reversal",
            "message": "애로호 사건의 수색 주체와 대상이 반대로 작성됨",
            "cap": 0.55,
        })

    # 3-2. 전쟁 선포 주체 오류
    # 모범답안: 영국이 전쟁을 선포
    # 사용자답안: 청나라가 전쟁을 선포
    model_has_britain_declared_war = (
        ("영국은" in model_norm or "영국이" in model_norm)
        and "전쟁을선포" in model_norm
    )

    user_has_qing_declared_war = (
        ("청나라는" in user_norm or "청나라가" in user_norm)
        and "전쟁을선포" in user_norm
    )

    if model_has_britain_declared_war and user_has_qing_declared_war:
        conflicts.append({
            "type": "wrong_war_declarer",
            "message": "전쟁 선포 주체를 영국이 아닌 청나라로 작성함",
            "cap": 0.55,
        })


    # 4. 반발 주체 오류
    model_has_britain_reaction = "영국의반발" in model_norm or "영국이반발" in model_norm
    user_has_qing_reaction = "청나라의반발" in user_norm or "청나라가반발" in user_norm

    if model_has_britain_reaction and user_has_qing_reaction:
        conflicts.append({
            "type": "wrong_actor_reaction",
            "message": "반발 주체를 영국이 아닌 청나라로 작성함",
            "cap": 0.55,
        })

    model_has_qing_reaction = "청나라의반발" in model_norm or "청나라가반발" in model_norm
    user_has_britain_reaction = "영국의반발" in user_norm or "영국이반발" in user_norm

    if model_has_qing_reaction and user_has_britain_reaction:
        conflicts.append({
            "type": "wrong_actor_reaction",
            "message": "반발 주체를 청나라가 아닌 영국으로 작성함",
            "cap": 0.55,
        })

    # 4-1. 국가/참전 주체 오류
    # 예: 모범답안은 프랑스 선교사 처형 사건과 프랑스의 가세인데,
    #     사용자 답안이 미국 선교사 처형 사건, 미국 가세라고 작성한 경우
    model_has_france_missionary = (
        "프랑스선교사" in model_norm
        or "프랑스신부" in model_norm
        or "프랑스가가세" in model_norm
        or "프랑스의가세" in model_norm
    )

    user_has_usa_missionary = (
        "미국선교사" in user_norm
        or "미국신부" in user_norm
        or "미국이가세" in user_norm
        or "미국의가세" in user_norm
    )

    if model_has_france_missionary and user_has_usa_missionary:
        conflicts.append({
            "type": "wrong_country_actor",
            "message": "프랑스 선교사 처형 사건과 프랑스의 가세를 미국으로 잘못 작성함",
            "cap": 0.55,
        })

    model_has_france_join = (
        "프랑스가가세" in model_norm
        or "프랑스의가세" in model_norm
        or "프랑스가참전" in model_norm
        or "프랑스의참전" in model_norm
    )

    user_has_usa_join = (
        "미국이가세" in user_norm
        or "미국의가세" in user_norm
        or "미국이참전" in user_norm
        or "미국의참전" in user_norm
    )

    if model_has_france_join and user_has_usa_join:
        conflicts.append({
            "type": "wrong_country_actor",
            "message": "프랑스의 가세 또는 참전을 미국으로 잘못 작성함",
            "cap": 0.55,
        })

    # 5. 문제의 핵심 사건명이 답안에 전혀 없는 경우
    # 단, 문제에 너무 일반적인 개념만 있으면 적용하지 않음
    strict_required_terms = [
        "애로호 사건",
        "임칙서",
        "프랑스 선교사 처형 사건",
        "난징조약",
    ]

    for term in strict_required_terms:
        if contains_any(question_text, [term]) and not contains_any(user_answer, [term]):
            # 프랑스 선교사 처형 사건은 표현 다양성을 고려
            if term == "프랑스 선교사 처형 사건":
                if contains_any(user_answer, ["프랑스 선교사", "프랑스 신부", "선교사 처형", "신부 처형"]):
                    continue

            conflicts.append({
                "type": "required_term_missing",
                "message": f"문제의 핵심 개념 '{term}'이 사용자 답안에 없음",
                "cap": 0.70,
            })

    return conflicts

def apply_factual_conflict_caps(final_score, conflicts):
    """사실 오류가 있을 경우 최종 점수 상한을 적용한다."""
    if not conflicts:
        return final_score

    cap = min(conflict.get("cap", 1.0) for conflict in conflicts)

    return min(final_score, cap)

def calculate_answer_length_score(model_answer, user_answer):
    """답안 길이의 적절성을 0~1 점수로 계산한다.

    짧은 서술형 답안을 기준으로 조정:
    - 너무 짧거나 무의미한 답안은 낮은 점수
    - 모범답안의 40~160% 정도 길이면 적절하다고 판단
    """
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

    grading_reference = (
        f"문제: {question_text}\n"
        f"모범 답안: {model_answer}\n\n"
        "사용자 답안이 위 문제와 모범 답안의 핵심 의미를 얼마나 충족하는지 판단하라."
    )

    try:
        tokenizer, reranker = get_reranker_model()

        inputs = tokenizer(
            grading_reference,
            user_answer,
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LENGTH,
            return_tensors="pt",
        )

        device = next(reranker.parameters()).device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = reranker(**inputs)
            logits = outputs.logits

        if logits.dim() == 2 and logits.size(-1) >= 2:
            probabilities = torch.softmax(logits[0], dim=-1)
            score = probabilities[-1].item()
        else:
            score = sigmoid(logits.view(-1)[0].item())

        score = max(0.0, min(float(score), 1.0))

        return round(score, 4)

    except Exception as e:
        print("[Reranker 채점 에러]", e)
        return None

def calculate_grading_score(question_text, model_answer, user_answer):
    """Reranker + 키워드 + 답안 길이 + 사실 오류 감지를 반영한 최종 자동 채점 점수.

    채점 기준:
    - Reranker 의미 유사도: 40%
    - 모범답안 핵심 키워드 포함률: 35%
    - 문제 핵심어 포함률: 15%
    - 답안 길이 적절성: 10%

    추가 제한:
    - 무의미한 답안은 0점
    - 핵심 키워드가 부족하면 고득점 제한
    - 차수 오류, 사건명 오류, 주체/대상 반전 오류가 있으면 점수 상한 적용
    """
    question_text = (question_text or "").strip()
    model_answer = (model_answer or "").strip()
    user_answer = (user_answer or "").strip()

    if not user_answer:
        return 0.0

    if is_meaningless_answer(user_answer):
        print(f"[자동 채점] 무의미 답안 감지: user_answer={user_answer}")
        return 0.0

    normalized_model = normalize_for_fact_check(model_answer)
    normalized_user = normalize_for_fact_check(user_answer)

    if normalized_model and normalized_model == normalized_user:
        print("[자동 채점] 모범답안과 사용자 답안 완전 일치")
        return 1.0

    reranker_score = calculate_reranker_score(
        question_text=question_text,
        model_answer=model_answer,
        user_answer=user_answer,
    )

    keyword_score, matched_keywords = calculate_keyword_coverage(
        reference_text=model_answer,
        user_answer=user_answer,
    )

    question_keyword_score, matched_question_keywords = calculate_keyword_coverage(
        reference_text=question_text,
        user_answer=user_answer,
    )

    length_score = calculate_answer_length_score(
        model_answer=model_answer,
        user_answer=user_answer,
    )

    if reranker_score is None:
        final_score = (
            keyword_score * 0.70
            + question_keyword_score * 0.20
            + length_score * 0.10
        )
        used_reranker_score = 0.0
        score_mode = "fallback_keyword_factcheck"
    else:
        final_score = (
            reranker_score * 0.40
            + keyword_score * 0.35
            + question_keyword_score * 0.15
            + length_score * 0.10
        )
        used_reranker_score = reranker_score
        score_mode = "reranker_factcheck"

    user_len = len(user_answer)

    # 1. 답안 길이에 따른 점수 상한
    if user_len < 10:
        final_score = min(final_score, 0.20)
    elif user_len < 20:
        final_score = min(final_score, 0.45)
    elif user_len < 35:
        final_score = min(final_score, 0.70)

    # 2. 모범답안 핵심 키워드 부족 시 상한
    if keyword_score < 0.20:
        final_score = min(final_score, 0.50)
    elif keyword_score < 0.40:
        final_score = min(final_score, 0.70)
    elif keyword_score < 0.60:
        final_score = min(final_score, 0.89)

    # 3. 문제에서 요구한 핵심어가 거의 없으면 정답 판정 불가
    if question_keyword_score < 0.20:
        final_score = min(final_score, 0.89)

    # 4. Reranker가 낮으면 키워드만으로 정답 처리되지 않도록 제한
    if reranker_score is not None and reranker_score < 0.50:
        final_score = min(final_score, 0.70)

    # 5. 핵심 사실 오류 감지 및 점수 상한 적용
    factual_conflicts = detect_factual_conflicts(
        question_text=question_text,
        model_answer=model_answer,
        user_answer=user_answer,
    )

    final_score = apply_factual_conflict_caps(
        final_score=final_score,
        conflicts=factual_conflicts,
    )

    final_score = max(0.0, min(final_score, 1.0))

    conflict_messages = [
        conflict.get("message", "")
        for conflict in factual_conflicts
        if conflict.get("message")
    ]

    print(
        f"[자동 채점] mode={score_mode}, "
        f"Reranker={used_reranker_score:.4f}, "
        f"모범키워드={keyword_score:.4f}, "
        f"문제키워드={question_keyword_score:.4f}, "
        f"길이={length_score:.4f}, "
        f"최종={final_score:.4f}, "
        f"매칭키워드={matched_keywords}, "
        f"문제매칭키워드={matched_question_keywords}, "
        f"사실오류={conflict_messages}"
    )

    return round(float(final_score), 4)

def predict_answer_label(grading_score):
    """자동 채점 점수를 기반으로 평가 라벨을 반환한다.

    기준:
    - 90점 이상: 정답
    - 60점 이상 90점 미만: 부분 정답
    - 60점 미만: 오답
    """
    if grading_score >= 0.90:
        return "정답"

    if grading_score >= 0.60:
        return "부분 정답"

    return "오답"
