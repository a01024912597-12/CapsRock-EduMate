import re

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.http import Http404
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from .answer_groups import SUBJECTIVE_QUIZ_TYPES, group_answer_rows, parse_human_score, quiz_review
from .lecture_thumbnail import capture_lecture_video_thumbnail
from .models import Classroom, ClassroomMembership, Lecture, Quiz, QuizAnswer, CLASSROOM_CODE_LENGTH
from .permissions import can_view_lecture, is_classroom_member, is_instructor
from .utils.option_utils import (
    normalize_subject_code,
    normalize_summary_api,
    normalize_whisper_model_name,
    normalize_worker_count,
)
from .utils.youtube_utils import extract_youtube_video_id

CLASSROOM_NAME_MAX_LENGTH = Classroom._meta.get_field("name").max_length
CLASSROOM_DESCRIPTION_MAX_LENGTH = Classroom._meta.get_field("description").max_length
LECTURE_TITLE_MAX_LENGTH = 150
LECTURE_DESCRIPTION_MAX_LENGTH = Lecture._meta.get_field("description").max_length

SUBJECT_TAG_RE = r"\s*\[\s*SUB\s*:\s*(?:auto|[1-4])\s*\]"
MY_GRADED_ROW_LIMIT = 10
MY_GRADED_ANSWER_SCAN = 300
GRADING_PAGE_SIZE = 15


def user_classrooms(user):
    """사용자가 만들었거나 참여 중인 강의실 목록."""
    return (
        Classroom.objects
        .filter(Q(owner=user) | Q(memberships__user=user))
        .distinct()
        .order_by("name", "pk")
    )


def _display_name(user):
    name = ""
    try:
        name = (user.profile.name or "").strip()
    except Exception:
        pass
    return name or user.get_full_name().strip() or user.username


def _normalize_code(raw):
    return "".join((raw or "").split()).replace("-", "").upper()


@login_required(login_url="home")
def classroom_create(request):
    if not is_instructor(request.user):
        messages.error(request, _("강의실은 강의자만 만들 수 있습니다."))
        return redirect("home")

    form = {"name": "", "description": ""}
    if request.method == "POST":
        form["name"] = (request.POST.get("name") or "").strip()
        form["description"] = (request.POST.get("description") or "").strip()

        if not form["name"]:
            messages.error(request, _("강의실 이름을 입력해 주세요."))
        elif len(form["name"]) > CLASSROOM_NAME_MAX_LENGTH:
            messages.error(request, _("강의실 이름은 %(max)d자 이하로 입력해 주세요.") % {"max": CLASSROOM_NAME_MAX_LENGTH})
        elif len(form["description"]) > CLASSROOM_DESCRIPTION_MAX_LENGTH:
            messages.error(request, _("강의실 설명은 %(max)d자 이하로 입력해 주세요.") % {"max": CLASSROOM_DESCRIPTION_MAX_LENGTH})
        else:
            classroom = Classroom.objects.create(
                owner=request.user,
                name=form["name"],
                description=form["description"],
            )
            messages.success(
                request,
                _("강의실을 만들었습니다. 학생들에게 입장 코드 %(code)s를 공유해 주세요.") % {"code": classroom.code},
            )
            return redirect("classroom_detail", pk=classroom.pk)

    return render(request, "classrooms/create.html", {
        "form": form,
        "name_max_length": CLASSROOM_NAME_MAX_LENGTH,
        "description_max_length": CLASSROOM_DESCRIPTION_MAX_LENGTH,
    })


@login_required(login_url="home")
def classroom_join(request):
    code = ""
    if request.method == "POST":
        code = _normalize_code(request.POST.get("code"))
        classroom = Classroom.objects.filter(code=code).first() if code else None

        if not code:
            messages.error(request, _("입장 코드를 입력해 주세요."))
        elif classroom is None:
            messages.error(request, _("입장 코드와 일치하는 강의실이 없습니다. 코드를 다시 확인해 주세요."))
        elif classroom.owner_id == request.user.id:
            messages.info(request, _("내가 만든 강의실입니다."))
            return redirect("classroom_detail", pk=classroom.pk)
        else:
            _membership, created = ClassroomMembership.objects.get_or_create(
                classroom=classroom,
                user=request.user,
            )
            if created:
                messages.success(request, _("%(name)s 강의실에 입장했습니다.") % {"name": classroom.name})
            else:
                messages.info(request, _("이미 참여 중인 강의실입니다."))
            return redirect("classroom_detail", pk=classroom.pk)

    return render(request, "classrooms/join.html", {
        "code": code,
        "code_length": CLASSROOM_CODE_LENGTH,
    })


@login_required(login_url="home")
def classroom_detail(request, pk):
    classroom = get_object_or_404(Classroom.objects.select_related("owner"), pk=pk)
    is_owner = classroom.owner_id == request.user.id
    is_member = classroom.memberships.filter(user=request.user).exists()

    if not (is_owner or is_member or request.user.is_staff):
        messages.error(request, _("참여 중인 강의실이 아닙니다. 입장 코드를 입력해 입장해 주세요."))
        return redirect("classroom_join")

    memberships = classroom.memberships.select_related("user", "user__profile")
    members = [
        {
            "user_id": m.user_id,
            "username": m.user.username,
            "display_name": _display_name(m.user),
            "joined_at": m.joined_at,
            "is_self": m.user_id == request.user.id,
        }
        for m in memberships
    ]

    posts = list(classroom_posts(classroom).order_by("-created_at"))
    member_ids = [m["user_id"] for m in members]

    if is_owner:
        summarized_counts = dict(
            Lecture.objects
            .filter(source_lecture__in=posts, user_id__in=member_ids, status=Lecture.STATUS_COMPLETED)
            .values("source_lecture_id")
            .annotate(n=Count("id"))
            .values_list("source_lecture_id", "n")
        )
        lectures = [
            _lecture_card(post, summarized_count=summarized_counts.get(post.id, 0))
            for post in posts
        ]
    else:
        # 학생은 강의자의 요약이 아닌 자기 사본의 상태를 본다.
        my_copies = {
            copy.source_lecture_id: copy
            for copy in Lecture.objects.filter(source_lecture__in=posts, user=request.user)
        }
        lectures = [_lecture_card(post, my_copy=my_copies.get(post.id)) for post in posts]

    my_graded_answers = []
    if is_member and not is_owner:
        from .views import _REVIEWED_Q

        my_graded_answers = group_answer_rows(
            QuizAnswer.objects
            .filter(
                _REVIEWED_Q,
                user=request.user,
                quiz_question__quiz__lecture__classroom=classroom,
                quiz_question__quiz__lecture__user=request.user,
            )
            .select_related("user", "quiz_question", "quiz_question__quiz", "quiz_question__quiz__lecture")
            .order_by("-created_at")[:MY_GRADED_ANSWER_SCAN]
        )[:MY_GRADED_ROW_LIMIT]
        for row in my_graded_answers:
            if row["kind"] == "single":
                row["lecture_title"] = _clean_title(row["obj"].quiz_question.quiz.lecture.title)
                continue
            row["lecture_title"] = _clean_title(row["lecture"].title)
            row["my_url"] = _my_quiz_result_url(row)
            if row["kind"] == "subjective":
                scores = list(
                    QuizAnswer.objects
                    .filter(user=request.user, quiz_question__quiz=row["quiz"], human_score__isnull=False)
                    .values_list("human_score", flat=True)
                )
                row["human_avg"] = round(sum(scores) / len(scores), 1) if scores else None

    return render(request, "classrooms/detail.html", {
        "classroom": classroom,
        "owner_name": _display_name(classroom.owner),
        "members": members,
        "member_count": len(members),
        "lectures": lectures,
        "is_owner": is_owner,
        "is_member": is_member,
        "my_graded_answers": my_graded_answers,
        "lecture_title_max_length": LECTURE_TITLE_MAX_LENGTH,
        "lecture_description_max_length": LECTURE_DESCRIPTION_MAX_LENGTH,
    })


def _my_quiz_result_url(row):
    """학생 본인이 그 차수 결과를 다시 볼 수 있는 화면."""
    lecture_id, generation = row["lecture"].id, row["quiz"].generation_number
    if row["kind"] == "objective":
        return f"{reverse('quiz_answer')}?lecture_id={lecture_id}&generation={generation}"
    return f"{reverse('feedback')}?lecture_id={lecture_id}&generation={generation}"


def classroom_posts(classroom):
    """강의자가 게시판에 올린 원본 강의 (학생 사본 제외)."""
    return classroom.lectures.filter(source_lecture__isnull=True, user_id=classroom.owner_id)


def _clean_title(title):
    return re.sub(SUBJECT_TAG_RE, "", title or "", flags=re.IGNORECASE).strip()


def _lecture_card(lecture, my_copy=None, summarized_count=None):
    thumbnail = lecture.thumbnail_url
    if not thumbnail and lecture.video_id:
        thumbnail = f"https://img.youtube.com/vi/{lecture.video_id}/0.jpg"
    if summarized_count is None:
        target = my_copy
        status = my_copy.status if my_copy else Lecture.STATUS_IDLE
    else:
        target = lecture
        status = lecture.status
    return {
        "id": lecture.id,
        "open_id": target.id if target else lecture.id,
        "title": _clean_title(lecture.title),
        "description": lecture.description,
        "thumbnail": thumbnail,
        "source_type": lecture.source_type,
        "status": status,
        "can_quiz": target is not None and status == Lecture.STATUS_COMPLETED and bool(target.summary_text),
        "created_at": lecture.created_at,
        "summarized_count": summarized_count,
    }


@login_required(login_url="home")
@require_POST
def classroom_kick(request, pk, user_id):
    classroom = get_object_or_404(Classroom, pk=pk, owner=request.user)
    membership = (
        ClassroomMembership.objects
        .select_related("user", "user__profile")
        .filter(classroom=classroom, user_id=user_id)
        .first()
    )
    if membership is None:
        messages.error(request, _("이미 강의실에 없는 학생입니다."))
    else:
        name = _display_name(membership.user)
        membership.delete()
        messages.success(request, _("%(name)s님을 강의실에서 추방했습니다.") % {"name": name})
    return redirect("classroom_detail", pk=classroom.pk)


@login_required(login_url="home")
@require_POST
def classroom_leave(request, pk):
    classroom = get_object_or_404(Classroom, pk=pk)
    if classroom.owner_id == request.user.id:
        messages.error(request, _("강의실을 만든 강의자는 나갈 수 없습니다."))
        return redirect("classroom_detail", pk=classroom.pk)

    ClassroomMembership.objects.filter(classroom=classroom, user=request.user).delete()
    messages.success(request, _("%(name)s 강의실에서 나갔습니다.") % {"name": classroom.name})
    return redirect("home")


@login_required(login_url="home")
@require_POST
def classroom_lecture_upload(request, pk):
    from .views import validate_lecture_video_file

    classroom = get_object_or_404(Classroom, pk=pk, owner=request.user)

    title = (request.POST.get("title") or "").strip()
    title = re.sub(SUBJECT_TAG_RE, "", title, flags=re.IGNORECASE).strip()
    description = (request.POST.get("description") or "").strip()
    upload_mode = (request.POST.get("upload_mode") or Lecture.SOURCE_YOUTUBE).strip()

    if not title:
        messages.error(request, _("강의 제목을 입력해 주세요."))
        return redirect("classroom_detail", pk=classroom.pk)
    if len(title) > LECTURE_TITLE_MAX_LENGTH:
        messages.error(request, _("강의 제목은 %(max)d자 이하로 입력해 주세요.") % {"max": LECTURE_TITLE_MAX_LENGTH})
        return redirect("classroom_detail", pk=classroom.pk)
    if len(description) > LECTURE_DESCRIPTION_MAX_LENGTH:
        messages.error(request, _("강의 설명은 %(max)d자 이하로 입력해 주세요.") % {"max": LECTURE_DESCRIPTION_MAX_LENGTH})
        return redirect("classroom_detail", pk=classroom.pk)

    final_title = f"{title} [SUB:auto]"

    if upload_mode == Lecture.SOURCE_FILE:
        lecture_file = request.FILES.get("lecture_file")
        file_error = validate_lecture_video_file(lecture_file)
        if file_error:
            messages.error(request, file_error)
            return redirect("classroom_detail", pk=classroom.pk)

        lecture = Lecture.objects.create(
            user=request.user,
            classroom=classroom,
            title=final_title,
            description=description,
            source_type=Lecture.SOURCE_FILE,
            video_file=lecture_file,
            status=Lecture.STATUS_IDLE,
        )
        thumbnail_url = capture_lecture_video_thumbnail(lecture)
        if thumbnail_url:
            lecture.thumbnail_url = thumbnail_url
            lecture.save(update_fields=["thumbnail_url"])
    else:
        youtube_link = (request.POST.get("youtube_link") or "").strip()
        video_id = extract_youtube_video_id(youtube_link) if youtube_link else ""
        if not video_id:
            messages.error(request, _("올바른 유튜브 링크를 입력해 주세요."))
            return redirect("classroom_detail", pk=classroom.pk)

        lecture = Lecture.objects.create(
            user=request.user,
            classroom=classroom,
            title=final_title,
            description=description,
            source_type=Lecture.SOURCE_YOUTUBE,
            youtube_url=youtube_link,
            video_id=video_id,
            thumbnail_url=f"https://img.youtube.com/vi/{video_id}/0.jpg",
            status=Lecture.STATUS_IDLE,
        )

    messages.success(request, _("강의를 올렸습니다. '요약하기'를 누르면 AI 요약이 시작됩니다."))
    return redirect("classroom_detail", pk=classroom.pk)


@login_required(login_url="home")
@require_POST
def lecture_summarize(request, lecture_id):
    """강의실 강의의 AI 요약을 시작한다.

    강의자는 자기 게시 강의를 요약하고, 학생은 자기 사본을 만들어 요약하므로
    요약 결과는 공유되지 않고 각자의 학습 기록에 남는다.
    """
    from .queue_worker import add_lecture_to_queue

    lecture = get_object_or_404(Lecture.objects.select_related("classroom"), pk=lecture_id)
    if lecture.classroom_id is None or not can_view_lecture(request.user, lecture):
        raise Http404

    if lecture.user_id != request.user.id:
        if not (lecture.is_classroom_post and is_classroom_member(request.user, lecture.classroom)):
            raise Http404
        lecture, _created = Lecture.objects.get_or_create(
            source_lecture=lecture,
            user=request.user,
            defaults={
                "classroom": lecture.classroom,
                "title": lecture.title,
                "description": lecture.description,
                "source_type": lecture.source_type,
                "youtube_url": lecture.youtube_url,
                "video_id": lecture.video_id,
                "video_file": lecture.video_file.name if lecture.video_file else "",
                "thumbnail_url": lecture.thumbnail_url,
                "status": Lecture.STATUS_IDLE,
            },
        )

    options = {
        "whisper_model": normalize_whisper_model_name(request.POST.get("whisper_model")),
        "workers": normalize_worker_count(request.POST.get("workers")),
        "subject_code": normalize_subject_code(request.POST.get("subject_code")),
        "summary_api": normalize_summary_api(request.POST.get("summary_api")),
    }
    base_title = re.sub(SUBJECT_TAG_RE, "", lecture.title or "", flags=re.IGNORECASE).strip()

    # 동시에 여러 명이 눌러도 한 번만 대기열에 들어가도록 상태 조건부로 갱신한다.
    started = Lecture.objects.filter(
        pk=lecture.pk,
        status__in=[Lecture.STATUS_IDLE, Lecture.STATUS_FAILED],
    ).update(
        status=Lecture.STATUS_PENDING,
        error_message="",
        analysis_requested_at=timezone.now(),
        title=f"{base_title} [SUB:{options['subject_code']}]",
    )

    if started:
        request.session[f"lecture_analysis_options_{lecture.id}"] = options
        request.session.modified = True
        add_lecture_to_queue(lecture.id, options)
    else:
        messages.info(request, _("이미 요약이 진행 중이거나 완료된 강의입니다."))

    return redirect(f"{reverse('summary')}?lecture_id={lecture.id}")


@login_required(login_url="home")
@require_POST
def classroom_lecture_remove(request, pk, lecture_id):
    classroom = get_object_or_404(Classroom, pk=pk, owner=request.user)
    updated = classroom_posts(classroom).filter(pk=lecture_id).update(classroom=None)
    if updated:
        messages.success(request, _("강의를 강의실에서 내렸습니다. 내 학습 기록에는 그대로 남아 있습니다."))
    else:
        messages.error(request, _("이미 강의실에 없는 강의입니다."))
    return redirect("classroom_detail", pk=classroom.pk)


def _quiz_type_label(quiz_type):
    return {
        "objective": _("객관식"),
        "tf": _("O/X"),
        "feedback": _("피드백"),
    }.get(quiz_type, _("서술형"))


@login_required(login_url="home")
def classroom_manage(request, pk):
    """강의자 전용 학습 관리: 학생별 학습 현황과 문제 풀이 채점."""
    from .views import HUMAN_LABEL_CHOICES, _REVIEWED_Q

    classroom = get_object_or_404(Classroom.objects.select_related("owner"), pk=pk, owner=request.user)

    memberships = list(classroom.memberships.select_related("user", "user__profile"))
    member_ids = [m.user_id for m in memberships]

    member_lectures = Lecture.objects.filter(
        classroom=classroom,
        source_lecture__isnull=False,
        user_id__in=member_ids,
    )
    member_answers = QuizAnswer.objects.filter(
        quiz_question__quiz__lecture__in=member_lectures,
        user_id__in=member_ids,
    )

    if request.method == "POST":
        answer = get_object_or_404(member_answers.select_related("user"), id=request.POST.get("answer_id"))
        human_label = (request.POST.get("human_label") or "").strip()
        human_score, score_ok = parse_human_score(request.POST.get("human_score"))

        if human_label not in HUMAN_LABEL_CHOICES:
            messages.error(request, _("채점 결과를 다시 선택해 주세요."))
        elif not score_ok:
            messages.error(request, _("점수는 0~100 사이의 숫자로 입력해 주세요."))
        else:
            answer.human_label = human_label
            answer.human_score = human_score
            answer.save(update_fields=["human_label", "human_score"])
            messages.success(
                request,
                _("%(name)s님의 답안 채점 결과를 저장했습니다.") % {"name": _display_name(answer.user)},
            )

        url = reverse("classroom_manage", args=[classroom.pk])
        querystring = request.POST.get("querystring", "")
        return redirect(f"{url}?{querystring}" if querystring else url)

    lecture_stats = {
        row["user_id"]: row
        for row in member_lectures.values("user_id").annotate(
            started=Count("id"),
            completed=Count("id", filter=Q(status=Lecture.STATUS_COMPLETED)),
        )
    }
    answer_stats = {
        row["user_id"]: row
        for row in member_answers.values("user_id").annotate(
            answered=Count("id"),
            reviewed=Count("id", filter=_REVIEWED_Q),
            auto_avg=Avg("similarity_score"),
            human_avg=Avg("human_score"),
        )
    }

    students = []
    for m in memberships:
        lec = lecture_stats.get(m.user_id, {})
        ans = answer_stats.get(m.user_id, {})
        answered = ans.get("answered", 0)
        reviewed = ans.get("reviewed", 0)
        auto_avg = ans.get("auto_avg")
        human_avg = ans.get("human_avg")
        students.append({
            "user_id": m.user_id,
            "display_name": _display_name(m.user),
            "username": m.user.username,
            "started": lec.get("started", 0),
            "completed": lec.get("completed", 0),
            "answered": answered,
            "reviewed": reviewed,
            "pending": answered - reviewed,
            "auto_avg": round(max(0.0, min(auto_avg, 1.0)) * 100, 1) if auto_avg is not None else None,
            "human_avg": round(human_avg, 1) if human_avg is not None else None,
        })

    posts = list(classroom_posts(classroom).order_by("-created_at").values("id", "title"))
    for post in posts:
        post["title"] = _clean_title(post["title"])

    student_id = (request.GET.get("student") or "").strip()
    post_id = (request.GET.get("lecture") or "").strip()
    review_state = (request.GET.get("review") or "").strip()

    answers = member_answers.select_related(
        "user", "user__profile", "quiz_question", "quiz_question__quiz", "quiz_question__quiz__lecture",
    )
    if student_id.isdigit():
        answers = answers.filter(user_id=int(student_id))
    if post_id.isdigit():
        answers = answers.filter(quiz_question__quiz__lecture__source_lecture_id=int(post_id))
    if review_state == "reviewed":
        answers = answers.filter(_REVIEWED_Q)
    elif review_state == "pending":
        answers = answers.exclude(_REVIEWED_Q)

    rows = group_answer_rows(answers.order_by("-created_at"))
    page_obj = Paginator(rows, GRADING_PAGE_SIZE).get_page(request.GET.get("page"))
    items = []
    for row in page_obj:
        if row["kind"] != "single":
            row["student_name"] = _display_name(row["user"])
            row["lecture_title"] = _clean_title(row["lecture"].title)
        else:
            answer = row["obj"]
            raw_score = answer.similarity_score if answer.similarity_score is not None else 0.0
            quiz = answer.quiz_question.quiz
            row.update({
                "student_name": _display_name(answer.user),
                "lecture_title": _clean_title(quiz.lecture.title),
                "quiz_type": _quiz_type_label(quiz.explanation),
                "percent": round(max(0.0, min(raw_score, 1.0)) * 100, 1),
                "is_reviewed": bool(answer.human_label),
            })
        items.append(row)

    params = request.GET.copy()
    params.pop("page", None)

    answer_total = member_answers.count()
    reviewed_total = member_answers.filter(_REVIEWED_Q).count()

    return render(request, "classrooms/manage.html", {
        "classroom": classroom,
        "students": students,
        "posts": posts,
        "items": items,
        "page_obj": page_obj,
        "student_id": student_id,
        "post_id": post_id,
        "review_state": review_state,
        "human_labels": HUMAN_LABEL_CHOICES,
        "base_query": params.urlencode(),
        "current_query": request.GET.urlencode(),
        "answer_total": answer_total,
        "reviewed_total": reviewed_total,
        "pending_total": answer_total - reviewed_total,
    })


@login_required(login_url="home")
def classroom_quiz_answers(request, pk, quiz_id, user_id):
    """강의자 전용: 학생 한 명이 푼 객관식/서술형 한 차수 전체를 보고 채점한다."""
    from .services.objective_quiz_service import OBJECTIVE_QUIZ_TYPE

    classroom = get_object_or_404(Classroom, pk=pk, owner=request.user)
    membership = get_object_or_404(
        ClassroomMembership.objects.select_related("user", "user__profile"),
        classroom=classroom,
        user_id=user_id,
    )
    quiz = get_object_or_404(
        Quiz.objects.select_related("lecture"),
        id=quiz_id,
        explanation__in=(OBJECTIVE_QUIZ_TYPE, *SUBJECTIVE_QUIZ_TYPES),
        lecture__classroom=classroom,
        lecture__source_lecture__isnull=False,
        lecture__user_id=user_id,
    )
    return quiz_review(
        request,
        quiz,
        membership.user,
        back_url=reverse("classroom_manage", args=[classroom.pk]),
        back_label=_("학습 관리로"),
        student_name=_display_name(membership.user),
        lecture_title=_clean_title(quiz.lecture.title),
    )
