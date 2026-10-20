"""학습 관리 화면에서 객관식·서술형 답안을 '학생 × 차수' 단위로 묶어 보여 주고 채점하는 공통 로직."""
from django.contrib import messages
from django.db.models import Count
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from .models import QuizAnswer, QuizQuestion
from .services.objective_quiz_service import FEEDBACK_QUIZ_TYPE, OBJECTIVE_QUIZ_TYPE, get_objective_meta

HUMAN_SCORE_MAX = 100.0
LABEL_CORRECT = "정답"
LABEL_PARTIAL = "부분 정답"
LABEL_WRONG = "오답"

KIND_OBJECTIVE = "objective"
KIND_SUBJECTIVE = "subjective"
# 예전 서술형 퀴즈는 유형 값이 비어 있다.
SUBJECTIVE_QUIZ_TYPES = ("", FEEDBACK_QUIZ_TYPE)

# 자동 판정 라벨(예전/현재 형식) → 강의자 채점 라벨
PREDICTED_TO_HUMAN_LABEL = {
    "정답 가능성 높음": LABEL_CORRECT,
    "검토 필요": LABEL_PARTIAL,
    "오답 가능성 높음": LABEL_WRONG,
    LABEL_CORRECT: LABEL_CORRECT,
    LABEL_PARTIAL: LABEL_PARTIAL,
    LABEL_WRONG: LABEL_WRONG,
}


def quiz_group_kind(quiz):
    """묶어서 보여 줄 퀴즈 유형. O/X 등은 None(답안별 한 줄)."""
    if quiz.explanation == OBJECTIVE_QUIZ_TYPE:
        return KIND_OBJECTIVE
    if quiz.explanation in SUBJECTIVE_QUIZ_TYPES:
        return KIND_SUBJECTIVE
    return None


def parse_human_score(raw):
    """(점수 또는 None, 유효 여부). 빈 값은 점수 없음으로 본다."""
    raw = (raw or "").strip()
    if not raw:
        return None, True
    try:
        score = float(raw)
    except ValueError:
        return None, False
    if not 0 <= score <= HUMAN_SCORE_MAX:
        return None, False
    return score, True


def objective_correct_choice(question):
    meta = get_objective_meta(question)
    return str(meta.get("correct_choice") or question.model_answer or "").strip()


def is_objective_correct(answer):
    choice = str(answer.user_answer or "").strip()
    return bool(choice) and choice == objective_correct_choice(answer.quiz_question)


def auto_percent(answer):
    if answer.similarity_score is None:
        return None
    return round(max(0.0, min(answer.similarity_score, 1.0)) * 100, 1)


def group_answer_rows(answers):
    """최신순 답안 목록을 행 목록으로 바꾼다.

    객관식·서술형은 (학생, 차수)마다 한 행으로 묶고, 그 밖의 유형은 답안마다 한 행이다.
    answers에는 user, quiz_question__quiz__lecture가 select_related 되어 있어야 한다.
    """
    rows = []
    groups = {}
    for answer in answers:
        quiz = answer.quiz_question.quiz
        kind = quiz_group_kind(quiz)
        if kind is None:
            rows.append({"kind": "single", "obj": answer, "created_at": answer.created_at})
            continue

        key = (answer.user_id, quiz.id)
        group = groups.get(key)
        if group is None:
            group = {
                "kind": kind,
                "user": answer.user,
                "quiz": quiz,
                "lecture": quiz.lecture,
                "created_at": answer.created_at,
                "answered": 0,
                "correct": 0,
                "reviewed": 0,
                "_auto_scores": [],
            }
            groups[key] = group
            rows.append(group)
        group["answered"] += 1
        group["reviewed"] += bool(answer.human_label)
        if kind == KIND_OBJECTIVE:
            group["correct"] += is_objective_correct(answer)
        else:
            percent = auto_percent(answer)
            if percent is not None:
                group["_auto_scores"].append(percent)

    if groups:
        question_counts = dict(
            QuizQuestion.objects
            .filter(quiz_id__in={quiz_id for _user_id, quiz_id in groups})
            .values("quiz_id")
            .annotate(n=Count("id"))
            .values_list("quiz_id", "n")
        )
        for (_user_id, quiz_id), group in groups.items():
            total = question_counts.get(quiz_id) or group["answered"]
            group["question_count"] = total
            if group["kind"] == KIND_OBJECTIVE:
                group["score"] = round(group["correct"] / total * 100, 1) if total else 0.0
            else:
                scores = group["_auto_scores"]
                group["score"] = round(sum(scores) / len(scores), 1) if scores else 0.0
            group["pending"] = group["answered"] - group["reviewed"]
            group["is_reviewed"] = group["pending"] == 0
    return rows


def _save_manual_grades(request, answers, student_name):
    from .views import HUMAN_LABEL_CHOICES

    updates = []
    for answer in answers:
        label = (request.POST.get(f"label_{answer.id}") or "").strip()
        score, score_ok = parse_human_score(request.POST.get(f"score_{answer.id}"))
        if label not in HUMAN_LABEL_CHOICES or not score_ok:
            messages.error(
                request,
                _("%(number)d번 문항의 채점 값이 올바르지 않습니다. 점수는 0~100 사이로 입력해 주세요.")
                % {"number": answer.quiz_question.number},
            )
            return
        answer.human_label = label
        answer.human_score = score
        updates.append(answer)
    QuizAnswer.objects.bulk_update(updates, ["human_label", "human_score"])
    messages.success(request, _("%(name)s님의 답안 채점 결과를 저장했습니다.") % {"name": student_name})


def _auto_grade(answer, kind):
    if kind == KIND_OBJECTIVE:
        correct = is_objective_correct(answer)
        return (LABEL_CORRECT if correct else LABEL_WRONG), (HUMAN_SCORE_MAX if correct else 0.0)
    return PREDICTED_TO_HUMAN_LABEL.get(answer.predicted_label, ""), auto_percent(answer)


def quiz_review(request, quiz, student, *, back_url, back_label, student_name, lecture_title):
    """한 학생이 푼 객관식/서술형 한 차수 전체를 목록으로 보여 주고 문제별·일괄 채점을 저장한다."""
    from .views import HUMAN_LABEL_CHOICES

    kind = quiz_group_kind(quiz)
    questions = list(quiz.questions.order_by("number", "id"))
    answers = {
        answer.quiz_question_id: answer
        for answer in QuizAnswer.objects.filter(user=student, quiz_question__quiz=quiz).select_related("quiz_question")
    }

    if request.method == "POST":
        if request.POST.get("action") == "auto":
            for answer in answers.values():
                answer.human_label, answer.human_score = _auto_grade(answer, kind)
            QuizAnswer.objects.bulk_update(answers.values(), ["human_label", "human_score"])
            messages.success(request, _("자동 채점 결과로 %(count)d문항을 채점했습니다.") % {"count": len(answers)})
        else:
            _save_manual_grades(request, answers.values(), student_name)
        return redirect(request.get_full_path())

    items = []
    correct_count = 0
    reviewed_count = 0
    auto_scores = []
    for question in questions:
        answer = answers.get(question.id)
        reviewed_count += bool(answer and answer.human_label)
        item = {"question": question, "answer": answer}
        if kind == KIND_OBJECTIVE:
            meta = get_objective_meta(question)
            correct_choice = objective_correct_choice(question)
            user_choice = str(answer.user_answer or "").strip() if answer else ""
            is_correct = bool(user_choice) and user_choice == correct_choice
            correct_count += is_correct
            item.update({
                "choices": [
                    {
                        "num": str(choice.get("c_num") or idx),
                        "text": choice.get("choice_text") or "",
                        "is_correct": str(choice.get("c_num") or idx) == correct_choice,
                        "is_selected": str(choice.get("c_num") or idx) == user_choice,
                    }
                    for idx, choice in enumerate((meta.get("choices") or [])[:4], start=1)
                ],
                "correct_choice": correct_choice,
                "user_choice": user_choice,
                "is_correct": is_correct,
                "explanation": meta.get("explanation") or "",
            })
        else:
            percent = auto_percent(answer) if answer else None
            if percent is not None:
                auto_scores.append(percent)
            item.update({
                "percent": percent,
                "auto_label": PREDICTED_TO_HUMAN_LABEL.get(answer.predicted_label, "") if answer else "",
                "keywords": question.explanation,
            })
        items.append(item)

    total = len(questions)
    if kind == KIND_OBJECTIVE:
        score = round(correct_count / total * 100, 1) if total else 0.0
    else:
        score = round(sum(auto_scores) / len(auto_scores), 1) if auto_scores else 0.0
    return render(request, "manage/quiz_answers.html", {
        "kind": kind,
        "quiz": quiz,
        "student": student,
        "student_name": student_name,
        "lecture_title": lecture_title,
        "items": items,
        "total_count": total,
        "answered_count": len(answers),
        "correct_count": correct_count,
        "reviewed_count": reviewed_count,
        "score": score,
        "human_labels": [label for label in HUMAN_LABEL_CHOICES if label],
        "back_url": back_url,
        "back_label": back_label,
    })
