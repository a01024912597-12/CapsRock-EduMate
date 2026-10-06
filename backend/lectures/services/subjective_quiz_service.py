import os
import re
import json

from django.conf import settings

from ..models import QuizQuestion, QuizAnswer
from .summary_service import gemini_generate_text, extract_concept_entries_from_summary
from .grading_service import calculate_grading_score, predict_answer_label

# "gemini"  : Gemini API로 서술형 문제 생성
# "template": Gemini 없이 요약문의 핵심 개념 사전을 기반으로 템플릿 문제 생성
QUIZ_GENERATION_MODE = getattr(
    settings,
    "QUIZ_GENERATION_MODE",
    os.environ.get("QUIZ_GENERATION_MODE", "gemini"),
)

_QUIZ_TIMESTAMP_RE = re.compile(
    r"\(\s*\d{1,3}\s*:\s*\d{1,2}(?:\s*:\s*\d{1,2})?\s*\)"
)

_QUIZ_KEYWORD_TIME_PAIR_RE = re.compile(
    r"([^()\n\r]+?)\s*(\(\s*\d{1,3}\s*:\s*\d{1,2}(?:\s*:\s*\d{1,2})?\s*\))"
)


def generate_quiz_with_gemini(summary_text, generation_number=1, previous_quiz_texts=""):
    """Gemini API로 짧은 서술형 예상문제 3개와 짧은 모범답안을 생성한다."""
    try:
        previous_instruction = ""

        if previous_quiz_texts:
            previous_instruction = (
                "\n\n이미 생성된 이전 차수의 문제는 아래와 같다. "
                "새 문제는 이전 문제와 최대한 겹치지 않게 출제하라.\n"
                f"{previous_quiz_texts}"
            )

        prompt = f"""너는 대학 강의 복습용 짧은 서술형 문제를 만드는 출제자다.
주어진 강의 요약을 바탕으로 짧은 서술형 예상 문제 3개와 각 문제의 짧은 모범 답안을 만들어라.

이번 출력은 {generation_number}차 예상 문제이다.

[문제 작성 조건]
- 문제는 반드시 1문장으로 작성할 것.
- 문제 길이는 35자 이내로 작성할 것.
- 한 문제는 핵심 개념 1개만 물을 것.
- 문제마다 서로 다른 핵심 개념을 다룰 것.
- 문제는 자동 채점이 가능하도록 명확하게 작성할 것.
- '전체 흐름', '배경과 결과를 모두'처럼 범위가 넓은 문제는 금지한다.

[모범 답안 작성 조건]
- 모범 답안은 반드시 2문장으로 작성할 것.
- 전체 길이는 120자 이내로 작성할 것.
- 첫 문장은 해당 개념의 의미를 직접 설명할 것.
- 둘째 문장은 강의에서 중요한 이유나 핵심 특징만 설명할 것.
- 핵심 키워드를 반드시 포함할 것.
- 불필요한 배경 설명과 여러 사건 나열은 금지한다.
- 모든 문장은 '~했다', '~이다', '~하다' 형식의 평어체로 작성할 것.

[타임라인 근거 작성 조건]
- 각 문제는 강의 요약의 '3. 꼭 알아야 할 필수 개념'에 있는 키워드와 (분:초)를 근거로 작성할 것.
- 관련 키워드에는 해당 문제와 직접 관련 있는 핵심 개념 1개만 작성할 것.
- 관련 타임라인에는 (분:초) 형식의 시간만 작성할 것.
- 요약문에서 근거를 찾을 수 없으면 관련 키워드와 관련 타임라인 모두 '근거 없음'이라고 작성할 것.

[출력 형식]
반드시 아래 형식을 그대로 지켜라.
'문제:', '관련 키워드:', '관련 타임라인:', '모범 답안:'이라는 표현을 반드시 그대로 사용하라.

1. 문제: ...
관련 키워드: ...
관련 타임라인: ...
모범 답안: ...

2. 문제: ...
관련 키워드: ...
관련 타임라인: ...
모범 답안: ...

3. 문제: ...
관련 키워드: ...
관련 타임라인: ...
모범 답안: ...

{previous_instruction}

강의 요약:
{summary_text}
"""

        result = gemini_generate_text(prompt)

        if not result:
            return "퀴즈 생성 중 오류가 발생했습니다."

        print(f"[Gemini 짧은 서술형 문제 생성] {generation_number}차 문제 생성 완료")
        return result

    except Exception as e:
        print("[Gemini 서술형 문제 생성 에러]", e)
        return "퀴즈 생성 중 오류가 발생했습니다."

def make_short_model_answer(keyword, description):
    """개념 설명을 기반으로 짧은 모범 답안을 만든다.

    채점 기준을 명확하게 하기 위해 2문장, 160자 이내로 제한한다.
    """
    keyword = (keyword or "").strip()
    description = (description or "").strip()

    if not description:
        return f"{keyword}는 강의에서 다룬 핵심 개념이다. 답안에는 {keyword}의 의미와 특징이 포함되어야 한다."

    parts = re.split(r"(?<=[.!?。다])\s+", description)
    parts = [
        part.strip(" -•\n\t")
        for part in parts
        if part.strip(" -•\n\t")
    ]

    if not parts:
        parts = [description]

    first_sentence = parts[0]

    if keyword and keyword not in first_sentence:
        first_sentence = f"{keyword}는 강의에서 다룬 핵심 개념이다."

    second_sentence = ""

    for part in parts:
        if part != first_sentence:
            second_sentence = part
            break

    if not second_sentence:
        second_sentence = f"답안에는 {keyword}의 의미와 핵심 특징이 포함되어야 한다."

    answer = f"{first_sentence} {second_sentence}".strip()

    if len(answer) > 160:
        answer = answer[:160].rstrip() + "."

    return answer

def build_template_question(keyword, index):
    """키워드와 번호를 바탕으로 짧고 명확한 서술형 문제를 만든다."""
    keyword = (keyword or "핵심 개념").strip()

    templates = [
        "'{keyword}'의 의미를 서술하시오.",
        "'{keyword}'의 핵심 특징을 서술하시오.",
        "'{keyword}'가 중요한 이유를 서술하시오.",
    ]

    template = templates[index % len(templates)]
    return template.format(keyword=keyword)

def generate_quiz_with_template(summary_text, generation_number=1, previous_quiz_texts=""):
    """Gemini API 없이 요약문 기반 템플릿 서술형 문제 3개를 생성한다."""
    entries = extract_concept_entries_from_summary(summary_text)

    if not entries:
        print("[템플릿 문제 생성] 핵심 개념 추출 실패, fallback 문제 생성")

        return """1. 문제: 강의에서 다룬 핵심 개념 하나를 선택하고 그 의미와 중요성을 서술하시오.
관련 키워드: 근거 없음
관련 타임라인: 근거 없음
모범 답안: 이 강의의 핵심 개념은 전체 내용을 이해하는 데 중요한 역할을 한다. 해당 개념의 의미와 중요성을 설명해야 한다.

2. 문제: 강의에서 나타난 중요한 사건의 원인과 결과를 서술하시오.
관련 키워드: 근거 없음
관련 타임라인: 근거 없음
모범 답안: 강의에서 다룬 중요한 사건은 특정한 배경과 원인 속에서 발생했다. 이 사건의 전개 과정과 결과를 함께 설명해야 한다.

3. 문제: 강의 전체 흐름에서 가장 중요하다고 생각되는 내용을 정리하고 그 이유를 서술하시오.
관련 키워드: 근거 없음
관련 타임라인: 근거 없음
모범 답안: 강의 전체 흐름에서 중요한 내용은 여러 개념과 사건을 연결해 이해하는 기준이 된다. 해당 내용을 중심으로 배경, 전개, 결과를 정리해야 한다.
"""

    start_index = ((generation_number - 1) * 3) % len(entries)

    selected_entries = []
    for i in range(min(3, len(entries))):
        selected_entries.append(entries[(start_index + i) % len(entries)])

    while len(selected_entries) < 3:
        selected_entries.append(entries[len(selected_entries) % len(entries)])

    quiz_blocks = []

    for idx, entry in enumerate(selected_entries, start=1):
        keyword = entry["keyword"]
        timeline = entry["timeline"]
        description = entry["description"]

        question = build_template_question(keyword, idx - 1)
        answer = make_short_model_answer(keyword, description)

        block = f"""{idx}. 문제: {question}
관련 키워드: {keyword}
관련 타임라인: {timeline}
모범 답안: {answer}"""

        quiz_blocks.append(block)

    quiz_text = "\n\n".join(quiz_blocks)

    print(f"[템플릿 서술형 문제 생성] {generation_number}차 문제 생성 완료")
    print(quiz_text)

    return quiz_text

def generate_quiz(summary_text, generation_number=1, previous_quiz_texts=""):
    """설정값에 따라 Gemini 방식 또는 템플릿 방식으로 서술형 문제를 생성한다."""
    mode = (QUIZ_GENERATION_MODE or "gemini").lower().strip()

    print(f"[퀴즈 생성 모드 확인] QUIZ_GENERATION_MODE={mode}")

    if mode == "template":
        return generate_quiz_with_template(
            summary_text=summary_text,
            generation_number=generation_number,
            previous_quiz_texts=previous_quiz_texts,
        )

    return generate_quiz_with_gemini(
        summary_text=summary_text,
        generation_number=generation_number,
        previous_quiz_texts=previous_quiz_texts,
    )

def split_quiz_reference(keywords_text="", timeline_text="", combined=""):
    """관련 키워드와 타임라인 문자열을 화면 표시용으로 분리한다."""
    keywords_text = (keywords_text or "").strip()
    timeline_text = (timeline_text or "").strip()
    combined = (combined or "").strip()

    if keywords_text and timeline_text:
        return keywords_text, timeline_text

    raw = combined or timeline_text or keywords_text

    if not raw or raw == "근거 없음":
        return "", raw

    timelines = [
        match.group(0).strip()
        for match in _QUIZ_TIMESTAMP_RE.finditer(raw)
    ]
    timelines_str = " / ".join(timelines)

    keywords = []

    for match in _QUIZ_KEYWORD_TIME_PAIR_RE.finditer(raw):
        keyword = match.group(1).strip().strip("·-,，、")

        if keyword and keyword not in keywords:
            keywords.append(keyword)

    if keywords:
        keywords_str = " / ".join(keywords)
    else:
        keywords_str = _QUIZ_TIMESTAMP_RE.sub("", raw)
        keywords_str = re.sub(r"[,，、\s]+", ", ", keywords_str).strip(" ,，、")

    if keywords_text:
        keywords_str = keywords_text

    if timeline_text:
        timelines_str = timeline_text

    return keywords_str, timelines_str

def _parse_quiz_block_body(rest):
    """한 문제 블록에서 문제, 관련 키워드, 관련 타임라인, 모범 답안을 분리한다."""
    answer_match = re.search(r"(?m)^\s*\**모범\s*답안\**\s*:\s*", rest)

    if not answer_match:
        return None

    before_answer = rest[:answer_match.start()]
    answer = rest[answer_match.end():].strip()

    keyword_match = re.search(
        r"(?m)^\s*관련\s*키워드\s*:\s*([^\r\n]+)\s*$",
        before_answer,
    )
    timeline_match = re.search(
        r"(?m)^\s*관련\s*타임라인\s*:\s*([^\r\n]+)\s*$",
        before_answer,
    )

    keywords = keyword_match.group(1).strip() if keyword_match else ""
    timeline = timeline_match.group(1).strip() if timeline_match else ""

    question = before_answer
    question = re.sub(r"(?m)^\s*관련\s*키워드\s*:.*$", "", question)
    question = re.sub(r"(?m)^\s*관련\s*타임라인\s*:.*$", "", question)
    question = question.strip()

    if timeline and not keywords:
        keywords, timeline = split_quiz_reference(combined=timeline)

    return question, keywords, timeline, answer

def parse_quiz_text(quiz_text):
    """AI가 생성한 텍스트를 문제/관련 키워드/관련 타임라인/모범답안 구조로 파싱한다."""
    quiz_items = []

    if not (quiz_text or "").strip():
        return quiz_items

    blocks = re.split(
        r"(?m)^(?=\d+\.\s*\**문제\**\s*:\s*)",
        quiz_text.strip(),
    )

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        number_match = re.match(r"(\d+)\.\s*\**문제\**\s*:\s*", block)

        if not number_match:
            continue

        number = number_match.group(1)
        rest = block[number_match.end():]

        parsed = _parse_quiz_block_body(rest)

        if not parsed:
            continue

        question, keywords, timeline, answer = parsed

        quiz_items.append({
            "number": number.strip(),
            "question": question.strip(),
            "answer": answer.strip(),
            "keywords": keywords.strip(),
            "timeline": timeline.strip(),
            "explanation": "",
        })

    if quiz_items:
        return quiz_items

    # 기존 형식 fallback
    pattern = r"(\d+)\.\s*문제\s*:\s*(.*?)(?:\n|\r\n)\s*모범\s*답안\s*:\s*(.*?)(?=\n\s*\d+\.\s*문제\s*:|\Z)"
    matches = re.findall(pattern, quiz_text, re.DOTALL)

    for number, question, answer in matches:
        quiz_items.append({
            "number": number.strip(),
            "question": question.strip(),
            "answer": answer.strip(),
            "keywords": "",
            "timeline": "",
            "explanation": "",
        })

    return quiz_items

def save_quiz_questions_from_text(quiz, quiz_text):
    """Quiz.quiz_text를 파싱하여 QuizQuestion에 개별 문제로 저장한다."""
    quiz_items = parse_quiz_text(quiz_text)

    if not quiz_items:
        print("[퀴즈 저장] 파싱된 문제가 없어 QuizQuestion 저장을 건너뜁니다.")
        return

    QuizQuestion.objects.filter(quiz=quiz).delete()

    for item in quiz_items:
        try:
            number = int(item.get("number", 1))
        except (TypeError, ValueError):
            number = 1

        keywords = (item.get("keywords") or "").strip()
        timeline = (item.get("timeline") or "").strip()

        if timeline and not keywords:
            keywords, timeline = split_quiz_reference(combined=timeline)

        QuizQuestion.objects.create(
            quiz=quiz,
            number=number,
            question_text=item.get("question", ""),
            model_answer=item.get("answer", ""),
            explanation=keywords[:2000],
            related_timeline=timeline[:500],
        )

    print(f"[퀴즈 저장] {quiz.generation_number}차 문제 {len(quiz_items)}개를 QuizQuestion에 저장했습니다.")

def ensure_quiz_questions_exist(quiz):
    """기존 quiz_text만 있고 QuizQuestion이 없는 경우 개별 문제를 생성한다."""
    if not quiz.questions.exists() and quiz.quiz_text:
        save_quiz_questions_from_text(quiz, quiz.quiz_text)

def _user_answers_dict(quiz):
    """기존 호환용: Quiz.answer에 JSON으로 저장된 학습자 답안을 dict로 반환한다."""
    raw = (getattr(quiz, "answer", None) or "").strip()

    if not raw:
        return {}

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    return data

def get_quiz_answer_dict_from_db(quiz, user):
    """QuizAnswer 테이블에서 현재 사용자의 답안을 dict 형태로 가져온다."""
    answer_dict = {}

    answers = (
        QuizAnswer.objects
        .filter(user=user, quiz_question__quiz=quiz)
        .select_related("quiz_question")
    )

    for answer in answers:
        answer_dict[str(answer.quiz_question.number)] = answer.user_answer

    return answer_dict

def save_user_answers_to_quiz_answer(user, quiz, post_data):
    """POST로 넘어온 사용자 답안을 QuizAnswer 테이블에 저장한다."""
    ensure_quiz_questions_exist(quiz)

    questions = quiz.questions.order_by("number")

    if not questions.exists():
        print("[답안 저장] QuizQuestion이 없어 QuizAnswer 저장을 건너뜁니다.")
        return {}

    question_map = {
        str(question.number): question
        for question in questions
    }

    legacy_json_data = {}

    for key in post_data:
        if not key.startswith("user_answer_"):
            continue

        suffix = key[len("user_answer_"):]
        value = post_data.get(key, "")

        if not isinstance(value, str):
            value = str(value)

        value = value.strip()

        if suffix == "raw":
            if value:
                legacy_json_data["__raw__"] = value
            continue

        if not suffix.isdigit():
            continue

        legacy_json_data[suffix] = value

        quiz_question = question_map.get(suffix)

        if not quiz_question:
            continue

        grading_score = calculate_grading_score(
            question_text=quiz_question.question_text,
            model_answer=quiz_question.model_answer,
            user_answer=value,
        )
        predicted_label = predict_answer_label(grading_score)

        QuizAnswer.objects.update_or_create(
            user=user,
            quiz_question=quiz_question,
            defaults={
                "user_answer": value,
                "similarity_score": grading_score,
                "predicted_label": predicted_label,
            },
        )

        print(
            f"[답안 자동 채점] {quiz.generation_number}차 {quiz_question.number}번 "
            f"채점점수={grading_score}, 판정={predicted_label}"
        )

    # 기존 Quiz.answer JSON 구조도 호환용으로 유지
    quiz.answer = json.dumps(legacy_json_data, ensure_ascii=False)
    quiz.save(update_fields=["answer"])

    print(f"[답안 저장] {quiz.generation_number}차 사용자 답안을 QuizAnswer에 저장했습니다.")
    return legacy_json_data

def build_quiz_items_for_display(quiz_text, quiz, user=None):
    """퀴즈 화면 출력용 데이터 구성.

    QuizQuestion에 저장된 개별 문제를 우선 사용하고,
    사용자 답안은 QuizAnswer 테이블을 우선 사용한다.
    """
    ensure_quiz_questions_exist(quiz)

    legacy_answer = _user_answers_dict(quiz)
    db_answer = {}

    if user and user.is_authenticated:
        db_answer = get_quiz_answer_dict_from_db(quiz, user)

    items = []
    questions = quiz.questions.order_by("number")

    if questions.exists():
        for question in questions:
            num = str(question.number)

            saved_answer = db_answer.get(num)

            if saved_answer is None:
                saved_answer = legacy_answer.get(num, "") or ""

            keywords = (question.explanation or "").strip()
            timelines = (getattr(question, "related_timeline", "") or "").strip()

            if not keywords and timelines:
                keywords, timelines = split_quiz_reference(combined=timelines)

            items.append({
                "number": question.number,
                "question": question.question_text,
                "answer": question.model_answer,
                "explanation": question.explanation,
                "keywords": keywords,
                "timeline": timelines,
                "timelines": timelines,
                "saved_answer": saved_answer,
            })

        raw_saved = legacy_answer.get("__raw__", "") or ""
        return items, raw_saved

    parsed_items = parse_quiz_text(quiz_text)

    for item in parsed_items:
        num = str(item["number"]).strip()

        saved_answer = db_answer.get(num)

        if saved_answer is None:
            saved_answer = legacy_answer.get(num, "") or ""

        keywords = (item.get("keywords") or "").strip()
        timelines = (item.get("timeline") or "").strip()

        if not keywords and timelines:
            keywords, timelines = split_quiz_reference(combined=timelines)

        items.append({
            **item,
            "keywords": keywords,
            "timeline": timelines,
            "timelines": timelines,
            "saved_answer": saved_answer,
        })

    raw_saved = legacy_answer.get("__raw__", "") or ""
    return items, raw_saved
