import re
from collections import Counter

from kiwipiepy import Kiwi

from .objective_quiz_service import build_objective_items_for_display
from .summary_service import extract_concept_entries_from_summary, gemini_generate_text
from ..utils.text_utils import clean_summary_text

kiwi = Kiwi()


def get_wrong_objective_items_from_quiz(user, objective_quiz):
    """객관식 퀴즈에서 사용자가 틀린 문항 목록을 만든다.

    서술형 피드백 문제 생성을 위해
    객관식 문항 번호, 문제 본문, 선택지, 사용자 선택, 정답, 해설, 키워드, 타임라인을 함께 저장한다.
    """
    if not objective_quiz:
        return []

    items = build_objective_items_for_display(objective_quiz, user)
    wrong_items = []

    for item in items:
        user_choice = str(item.get("user_choice") or "").strip()
        correct_choice = str(item.get("correct_choice") or "").strip()

        is_correct = bool(user_choice) and user_choice == correct_choice

        if is_correct:
            continue

        selected_text = ""
        correct_text = ""

        choices = item.get("choices") or []

        for choice in choices:
            c_num = str(choice.get("c_num") or "").strip()
            choice_text = str(choice.get("choice_text") or "").strip()

            if c_num == user_choice:
                selected_text = choice_text

            if c_num == correct_choice:
                correct_text = choice_text

        wrong_items.append({
            "objective_number": item.get("q_num") or item.get("number"),
            "question_number": item.get("q_num") or item.get("number"),
            "question_text": item.get("question_text") or "",
            "choices": choices,
            "user_choice": user_choice or "미선택",
            "selected_text": selected_text or "미선택",
            "correct_choice": correct_choice,
            "correct_text": correct_text,
            "keyword": item.get("keyword") or correct_text or "핵심 개념",
            "timeline": item.get("timeline") or "근거 없음",
            "explanation": item.get("explanation") or "",
        })

    return wrong_items

def _has_valid_timeline(value):
    """타임라인 근거가 있는지 확인한다."""
    value = (value or "").strip()

    if not value or value == "근거 없음":
        return False

    return bool(re.search(r"\d{1,3}\s*:\s*\d{1,2}", value))

def rank_wrong_items_for_feedback(wrong_items, summary_text=""):
    """오답 문항을 서술형 피드백 생성 우선순위에 따라 정렬한다.

    우선순위:
    1. 요약 핵심 개념과 연결된 오답
    2. 관련 타임라인이 있는 오답
    3. 정답 선택지 또는 키워드가 명확한 오답
    4. 해설이 있는 오답
    5. 객관식 번호가 앞선 오답
    """
    summary_text = clean_summary_text(summary_text or "")
    concept_entries = extract_concept_entries_from_summary(summary_text)

    concept_keywords = []

    for entry in concept_entries:
        keyword = (entry.get("keyword") or "").strip()

        if keyword and keyword not in concept_keywords:
            concept_keywords.append(keyword)

    def score_item(item):
        score = 0

        objective_number = item.get("objective_number") or item.get("question_number") or 9999

        try:
            objective_number = int(objective_number)
        except (TypeError, ValueError):
            objective_number = 9999

        keyword = (item.get("keyword") or "").strip()
        correct_text = (item.get("correct_text") or "").strip()
        timeline = (item.get("timeline") or "").strip()
        explanation = (item.get("explanation") or "").strip()

        keyword_source = f"{keyword} {correct_text}"

        # 1순위: 요약의 핵심 개념과 연결된 오답
        for concept_keyword in concept_keywords:
            if concept_keyword and (
                concept_keyword in keyword_source
                or keyword in concept_keyword
                or correct_text in concept_keyword
            ):
                score += 100
                break

        # 2순위: 타임라인 근거가 있는 오답
        if _has_valid_timeline(timeline):
            score += 30

        # 3순위: 정답 선택지 또는 키워드가 명확한 오답
        if keyword and keyword != "핵심 개념":
            score += 20

        if correct_text and correct_text not in ["미선택", ""]:
            score += 15

        # 4순위: 해설이 있는 오답
        if explanation:
            score += 10

        # 번호가 앞선 문항을 약간 우선
        score -= objective_number * 0.01

        return score

    ranked_items = sorted(
        wrong_items or [],
        key=score_item,
        reverse=True,
    )

    return ranked_items
FEEDBACK_BAD_KEYWORDS = {
    "다음", "내용", "보기", "정답", "문제", "설명", "개념", "강의", "관련", "해당",
}


def clean_choice_text_for_feedback(value):
    """객관식 선택지 문장에서 '1번', '①' 같은 표시를 제거한다."""
    value = (value or "").strip()

    value = re.sub(r"^\s*\d+\s*번\s*", "", value)
    value = re.sub(r"^\s*[①②③④⑤]\s*", "", value)
    value = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value

def is_bad_feedback_keyword(value):
    """서술형 문제 핵심 개념으로 쓰기 부적절한 단어인지 확인한다."""
    value = clean_choice_text_for_feedback(value)

    if not value:
        return True

    if value in FEEDBACK_BAD_KEYWORDS:
        return True

    # 한 글자는 무조건 부적절
    if len(value) <= 1:
        return True

    return False

def make_context_concept_from_blank_question(question_text, answer_word):
    """객관식 빈칸 문제 문맥을 이용해 핵심 개념을 보정한다.

    예:
    - '____전쟁' + '아편' -> '아편전쟁'
    - '아편____' + '전쟁' -> '아편전쟁'
    """
    question_text = (question_text or "").strip()
    answer_word = clean_choice_text_for_feedback(answer_word)

    if not question_text or not answer_word:
        return ""

    blank_patterns = [
        r"_{2,}",
        r"□+",
        r"\(\s*\)",
        r"\[\s*빈칸\s*\]",
        r"빈칸",
    ]

    blank_regex = "|".join(blank_patterns)

    match = re.search(
        rf"([가-힣A-Za-z0-9]{{0,6}})\s*(?:{blank_regex})\s*([가-힣A-Za-z0-9]{{0,6}})",
        question_text,
    )

    if not match:
        return ""

    prefix = (match.group(1) or "").strip()
    suffix = (match.group(2) or "").strip()

    # 조사나 너무 일반적인 접미 표현은 제거
    suffix = re.sub(r"^(은|는|이|가|을|를|의|에|에서|으로|로|와|과|하고)", "", suffix)
    prefix = re.sub(r"(은|는|이|가|을|를|의|에|에서|으로|로|와|과|하고)$", "", prefix)

    candidate = f"{prefix}{answer_word}{suffix}".strip()

    # 너무 길거나 이상한 후보는 버림
    if len(candidate) < 2 or len(candidate) > 15:
        return ""

    if is_bad_feedback_keyword(candidate):
        return ""

    return candidate

def choose_feedback_concept(item):
    """서술형 문제에 사용할 핵심 개념을 선택한다.

    우선순위:
    1. 객관식 빈칸 문맥으로 보정한 개념
    2. 정답 선택지
    3. 기존 keyword
    4. 해설에서 추출 가능한 명사
    """
    question_text = (item.get("question_text") or "").strip()
    raw_keyword = clean_choice_text_for_feedback(item.get("keyword") or "")
    correct_text = clean_choice_text_for_feedback(item.get("correct_text") or "")
    explanation = (item.get("explanation") or "").strip()

    # 1. keyword가 짧거나 일반 명사이면 빈칸 문맥으로 보정
    if raw_keyword:
        context_concept = make_context_concept_from_blank_question(
            question_text=question_text,
            answer_word=raw_keyword,
        )

        if context_concept:
            return context_concept

    # 2. 정답 선택지가 더 구체적이면 정답 선택지 사용
    if correct_text and not is_bad_feedback_keyword(correct_text):
        return correct_text

    # 3. 기존 keyword가 쓸 만하면 사용
    if raw_keyword and not is_bad_feedback_keyword(raw_keyword):
        return raw_keyword

    # 4. 해설에서 명사 후보 추출
    try:
        nouns = []

        for token in kiwi.tokenize(explanation):
            word = token.form.strip()

            if not token.tag.startswith("N"):
                continue

            if is_bad_feedback_keyword(word):
                continue

            if len(word) < 2:
                continue

            if word not in nouns:
                nouns.append(word)

        if nouns:
            return nouns[0]

    except Exception as e:
        print("[피드백 개념 추출 에러]", e)

    # 5. 마지막 fallback
    if correct_text:
        return correct_text

    if raw_keyword:
        return raw_keyword

    return "핵심 개념"

def build_feedback_question_by_context(keyword, explanation="", question_text="", index=1, variant=1):
    """오답 개념과 해설 문맥을 바탕으로 서술형 피드백 문제를 만든다.

    같은 개념이 여러 번 선택된 경우 variant 값에 따라 질문 방향을 다르게 만든다.
    """
    keyword = (keyword or "핵심 개념").strip()
    explanation = (explanation or "").strip()
    question_text = (question_text or "").strip()

    context = f"{question_text} {explanation}"

    if "애로호" in keyword or "애로호" in context:
        questions = [
            "애로호 사건이 제2차 아편전쟁의 직접적인 계기가 된 이유를 서술하시오.",
            "애로호 사건이 청나라와 영국의 갈등을 심화시킨 이유를 서술하시오.",
            "애로호 사건이 외교 분쟁으로 확대된 배경을 서술하시오.",
        ]
        return questions[(variant - 1) % len(questions)]

    if "임칙서" in keyword or "임칙서" in context:
        questions = [
            "임칙서의 아편 단속이 청나라와 영국의 갈등에 끼친 영향을 서술하시오.",
            "임칙서가 아편 단속을 추진한 이유를 서술하시오.",
            "임칙서의 아편 단속이 아편전쟁과 연결된 과정을 서술하시오.",
        ]
        return questions[(variant - 1) % len(questions)]

    if "난징조약" in keyword or "난징조약" in context:
        questions = [
            "난징조약이 불평등 조약으로 평가되는 이유를 서술하시오.",
            "난징조약이 청나라에 끼친 영향을 서술하시오.",
            "난징조약의 핵심 내용을 서술하시오.",
        ]
        return questions[(variant - 1) % len(questions)]

    if "아편전쟁" in keyword or "아편전쟁" in context:
        questions = [
            "아편전쟁이 청나라와 영국 관계에 끼친 영향을 서술하시오.",
            "아편전쟁이 발생한 핵심 원인을 서술하시오.",
            "아편전쟁 이후 청나라가 겪은 변화를 서술하시오.",
        ]
        return questions[(variant - 1) % len(questions)]

    if "불평등" in keyword or "조약" in keyword or "불평등" in context:
        questions = [
            f"{keyword}의 핵심 특징을 서술하시오.",
            f"{keyword}가 중요한 이유를 서술하시오.",
            f"{keyword}가 끼친 영향을 서술하시오.",
        ]
        return questions[(variant - 1) % len(questions)]

    templates = [
        f"{keyword}의 핵심 의미를 서술하시오.",
        f"{keyword}가 강의에서 중요하게 다뤄진 이유를 서술하시오.",
        f"{keyword}의 핵심 특징을 서술하시오.",
    ]

    return templates[(variant - 1) % len(templates)]

def normalize_feedback_concept_key(concept):
    """서술형 피드백 문제의 중복 개념 판별용 key를 만든다.

    예:
    - '애로호 사건' -> '애로호사건'
    - "'애로호 사건'" -> '애로호사건'
    - '애로호 사건의 의미' -> '애로호사건'
    """
    concept = (concept or "").strip()

    concept = concept.replace("'", "").replace('"', "")
    concept = re.sub(r"\s+", "", concept)

    remove_words = [
        "의미", "핵심", "특징", "중요성", "중요한이유",
        "이유", "배경", "결과", "역할", "영향",
        "개념", "내용",
    ]

    protected_words = [
        "애로호사건",
        "아편전쟁",
        "난징조약",
        "임칙서의아편단속",
        "임칙서",
    ]

    if concept in protected_words:
        return concept

    for word in remove_words:
        if concept.endswith(word) and len(concept) > len(word) + 1:
            concept = concept[:-len(word)]

    return concept.strip()


def is_duplicate_feedback_concept(concept_key, seen_concept_keys):
    """이미 선택된 개념과 중복되는지 확인한다.

    완전 일치뿐 아니라,
    '애로호'와 '애로호사건'처럼 포함 관계도 중복으로 본다.
    """
    if not concept_key:
        return True

    for seen_key in seen_concept_keys:
        if not seen_key:
            continue

        if concept_key == seen_key:
            return True

        if concept_key in seen_key or seen_key in concept_key:
            return True

    return False

def build_feedback_quiz_text_from_wrong_items(wrong_items, generation_number=1, summary_text=""):
    """객관식 오답 문항을 기반으로 짧은 서술형 피드백 문제를 생성한다.

    개선 내용:
    - 오답 문항을 중요도 기준으로 정렬한다.
    - 서로 다른 핵심 개념을 우선 선택한다.
    - 문제가 3개보다 부족하면 중복 개념이라도 다른 질문 방향으로 보충한다.
    - 한 문제는 하나의 핵심 개념만 묻는다.
    - 문제와 모범 답안은 짧게 생성한다.
    - 화면 표시를 위해 '관련 키워드' 필드에는 '객관식 n번' 정보를 저장한다.
    """
    if not wrong_items:
        return """1. 문제: 강의에서 가장 중요한 핵심 개념 하나의 의미를 한두 문장으로 서술하시오.
관련 키워드: 객관식 오답 없음
관련 타임라인: 근거 없음
모범 답안: 핵심 개념은 강의 내용을 이해하는 데 필요한 중심 내용이다. 답안에는 해당 개념의 의미와 강의 속 역할이 포함되어야 한다.
"""

    ranked_wrong_items = rank_wrong_items_for_feedback(
        wrong_items=wrong_items,
        summary_text=summary_text,
    )

    # =========================
    # 중요도 기준 + 중복 완화 방식으로 최대 3개 선택
    # =========================
    target_count = min(3, len(ranked_wrong_items))

    selected_items = []
    selected_objective_numbers = set()
    seen_concept_keys = set()

    # 1차 선택: 서로 다른 핵심 개념을 우선 선택
    for item in ranked_wrong_items:
        concept = choose_feedback_concept(item)
        concept_key = normalize_feedback_concept_key(concept)
        objective_number = item.get("objective_number") or item.get("question_number")

        if objective_number in selected_objective_numbers:
            continue

        if is_duplicate_feedback_concept(concept_key, seen_concept_keys):
            print(f"[서술형 피드백 1차 중복 보류] concept={concept}, key={concept_key}")
            continue

        copied_item = dict(item)
        copied_item["_feedback_concept"] = concept
        copied_item["_feedback_variant"] = 1

        selected_items.append(copied_item)
        selected_objective_numbers.add(objective_number)
        seen_concept_keys.add(concept_key)

        if len(selected_items) >= target_count:
            break

    # 2차 보충: 문제가 부족하면 중복 개념이라도 다른 객관식 오답 문항을 추가
    if len(selected_items) < target_count:
        concept_count = {}

        for selected in selected_items:
            selected_concept = selected.get("_feedback_concept") or choose_feedback_concept(selected)
            selected_key = normalize_feedback_concept_key(selected_concept)
            concept_count[selected_key] = concept_count.get(selected_key, 0) + 1

        for item in ranked_wrong_items:
            if len(selected_items) >= target_count:
                break

            objective_number = item.get("objective_number") or item.get("question_number")

            if objective_number in selected_objective_numbers:
                continue

            concept = choose_feedback_concept(item)
            concept_key = normalize_feedback_concept_key(concept)

            next_variant = concept_count.get(concept_key, 0) + 1

            copied_item = dict(item)
            copied_item["_feedback_concept"] = concept
            copied_item["_feedback_variant"] = next_variant

            selected_items.append(copied_item)
            selected_objective_numbers.add(objective_number)
            concept_count[concept_key] = next_variant

            print(
                f"[서술형 피드백 중복 개념 보충] "
                f"concept={concept}, key={concept_key}, variant={next_variant}"
            )

    # 그래도 비어 있으면 최소 1개 생성
    if not selected_items and ranked_wrong_items:
        first_item = dict(ranked_wrong_items[0])
        first_item["_feedback_concept"] = choose_feedback_concept(first_item)
        first_item["_feedback_variant"] = 1
        selected_items = [first_item]

    def get_objective_number_for_sort(item):
        """관련 객관식 번호 기준 정렬용 함수."""
        number = item.get("objective_number") or item.get("question_number") or 9999

        try:
            return int(number)
        except (TypeError, ValueError):
            return 9999

    selected_items = sorted(
        selected_items,
        key=get_objective_number_for_sort,
    )
    wrong_context_blocks = []

    for index, item in enumerate(selected_items, start=1):
        objective_number = item.get("objective_number") or item.get("question_number") or index
        question_text = (item.get("question_text") or "").strip()
        user_choice = str(item.get("user_choice") or "미선택").strip()
        selected_text = (item.get("selected_text") or "미선택").strip()
        correct_choice = str(item.get("correct_choice") or "").strip()
        correct_text = (item.get("correct_text") or "").strip()
        explanation = (item.get("explanation") or "").strip()
        keyword = (item.get("_feedback_concept") or choose_feedback_concept(item)).strip()
        timeline = (item.get("timeline") or "근거 없음").strip()

        choices = item.get("choices") or []
        choices_text_list = []

        for choice in choices:
            c_num = str(choice.get("c_num") or "").strip()
            choice_text = str(choice.get("choice_text") or "").strip()

            if c_num and choice_text:
                choices_text_list.append(f"{c_num}. {choice_text}")

        choices_text = "\n".join(choices_text_list) if choices_text_list else "선택지 정보 없음"

        wrong_context_blocks.append(f"""[오답 문항 {index}]
객관식 원문 번호: {objective_number}

객관식 문제:
{question_text}

선택지:
{choices_text}

사용자 선택:
{user_choice}번 - {selected_text}

정답:
{correct_choice}번 - {correct_text}

정답 핵심 개념:
{keyword}

관련 타임라인:
{timeline}

객관식 해설:
{explanation if explanation else "해설 없음"}
""")

    wrong_context = "\n\n".join(wrong_context_blocks)

    prompt = f"""너는 대학 강의 복습용 서술형 피드백 문제를 만드는 출제자다.
아래는 학습자가 객관식 문제에서 틀린 문항 목록이다.

객관식 오답 문항을 바탕으로, 학습자가 틀린 핵심 개념을 짧게 설명하도록 서술형 피드백 문제를 생성하라.

[출제 목표]
- 객관식 오답 문항의 정답 핵심 개념을 중심으로 문제를 만들 것.
- 한 문제는 반드시 하나의 개념만 물을 것.
- 문제는 짧고 명확해야 하며, 자동 채점 기준이 흔들리지 않도록 만들 것.
- 사용자가 긴 글을 쓰지 않아도 답할 수 있도록 만들 것.
- 문제 수는 오답 문항 수에 맞추되 최대 3문항만 생성할 것.
- 같은 핵심 개념을 반복해서 출제하지 말 것.
- 단, 같은 개념의 오답 문항만 제공된 경우에는 서로 다른 관점의 문제로 출제할 것.

[문제 작성 조건]
- 문제는 반드시 1문장으로 작성할 것.
- 문제 길이는 50자 이내로 작성할 것.
- 문제에는 정답 핵심 개념을 반드시 포함할 것.
- 문제에 사용할 핵심 개념은 반드시 구체적인 개념명이어야 한다.
- '사건', '개념', '내용', '원인', '결과', '배경', '의미'처럼 단독으로 의미가 불명확한 일반 명사는 문제의 핵심 개념으로 사용하지 말 것.
- '아편'처럼 단독 단어만으로 범위가 애매한 경우, 객관식 문제 문맥을 참고하여 '아편전쟁'처럼 더 구체적인 개념명으로 바꿔 작성할 것.
- '배경과 결과를 모두 서술하시오', '전체 흐름을 서술하시오'처럼 범위가 넓은 문제는 만들지 말 것.

[좋은 문제 예시]
- 애로호 사건이 제2차 아편전쟁의 직접적인 계기가 된 이유를 서술하시오.
- 임칙서의 아편 단속이 청나라와 영국의 갈등에 끼친 영향을 서술하시오.
- 난징조약이 불평등 조약으로 평가되는 이유를 서술하시오.

[나쁜 문제 예시]
- 사건의 의미를 서술하시오.
- 아편의 의미를 서술하시오.
- 애로호 사건의 의미를 서술하시오.
- 애로호 사건이 중요한 이유를 서술하시오.

[모범 답안 작성 조건]
- 모범 답안은 반드시 2문장으로 작성할 것.
- 전체 길이는 140자 이내로 작성할 것.
- 첫 문장에는 문제에서 사용한 구체적인 핵심 개념명을 그대로 포함할 것.
- 첫 문장은 해당 개념의 의미를 직접 설명할 것.
- 둘째 문장은 강의에서 중요한 이유나 핵심 특징만 설명할 것.
- 불필요한 배경 설명, 긴 예시, 여러 사건 나열은 금지한다.
- 모든 문장은 '~했다', '~이다', '~하다' 형식의 평어체로 작성할 것.

[관련 정보 작성 조건]
- 관련 키워드에는 실제 키워드를 쓰지 말고 반드시 '객관식 n번' 형식으로 작성할 것.
- 예: 관련 키워드: 객관식 3번
- 관련 타임라인은 오답 문항에 제공된 값을 그대로 사용할 것.
- 타임라인이 없으면 '근거 없음'이라고 작성할 것.

[출력 형식]
반드시 아래 형식을 그대로 지켜라.
'문제:', '관련 키워드:', '관련 타임라인:', '모범 답안:'이라는 표현을 반드시 그대로 사용하라.

1. 문제: ...
관련 키워드: 객관식 n번
관련 타임라인: ...
모범 답안: ...

2. 문제: ...
관련 키워드: 객관식 n번
관련 타임라인: ...
모범 답안: ...

3. 문제: ...
관련 키워드: 객관식 n번
관련 타임라인: ...
모범 답안: ...

[객관식 오답 문항 목록]
{wrong_context}
"""

    try:
        result = gemini_generate_text(prompt)

        if result and result.strip():
            print(f"[오답 기반 짧은 서술형 피드백 생성] {generation_number}차 Gemini 생성 완료")
            return result.strip()

    except Exception as e:
        print("[오답 기반 서술형 피드백 Gemini 생성 에러]", e)

    # =========================
    # Gemini 실패 시 fallback
    # =========================
    quiz_blocks = []
    used_question_texts = set()

    for index, item in enumerate(selected_items, start=1):
        objective_number = item.get("objective_number") or item.get("question_number") or index
        keyword = (item.get("_feedback_concept") or choose_feedback_concept(item)).strip()
        timeline = (item.get("timeline") or "근거 없음").strip()
        explanation = (item.get("explanation") or "").strip()
        question_text = (item.get("question_text") or "").strip()
        variant = item.get("_feedback_variant", 1)

        question = build_feedback_question_by_context(
            keyword=keyword,
            explanation=explanation,
            question_text=question_text,
            index=index,
            variant=variant,
        )

        # 질문 문장이 완전히 중복되면 variant를 증가시켜 다시 생성
        retry_variant = variant
        while question in used_question_texts and retry_variant <= 5:
            retry_variant += 1
            question = build_feedback_question_by_context(
                keyword=keyword,
                explanation=explanation,
                question_text=question_text,
                index=index,
                variant=retry_variant,
            )

        used_question_texts.add(question)

        short_explanation = ""

        if explanation:
            parts = re.split(r"(?<=[.!?。다])\s+", explanation)
            parts = [
                part.strip()
                for part in parts
                if part.strip()
            ]

            if parts:
                short_explanation = parts[0]

        if short_explanation:
            model_answer = f"{keyword}는 객관식 {objective_number}번의 핵심 개념이다. {short_explanation}"
        else:
            model_answer = (
                f"{keyword}는 객관식 {objective_number}번의 정답 개념이다. "
                f"답안에는 {keyword}의 의미와 핵심 특징이 포함되어야 한다."
            )

        if len(model_answer) > 160:
            model_answer = model_answer[:160].rstrip() + "."

        block = f"""{index}. 문제: {question}
관련 키워드: 객관식 {objective_number}번
관련 타임라인: {timeline}
모범 답안: {model_answer}"""

        quiz_blocks.append(block)

    return "\n\n".join(quiz_blocks)
