import os
import re
import json
from datetime import timedelta, datetime

from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, FileResponse, HttpResponse, Http404, HttpResponseRedirect
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST
from django.utils.translation import check_for_language, gettext as _, gettext_lazy as _lazy
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Exists, OuterRef, Count, Q, Max, Avg, Value, Subquery
from django.db.models.functions import Coalesce
from django.utils import timezone, translation
from django.urls import reverse, NoReverseMatch
from django.core.paginator import Paginator

from .models import Lecture, UserProfile, Quiz, QuizQuestion, QuizAnswer, StudyCalendarMemo
from .lecture_thumbnail import capture_lecture_video_thumbnail
from .utils.youtube_utils import extract_youtube_video_id
from .utils.option_utils import (
    normalize_whisper_model_name,
    normalize_subject_code,
    normalize_worker_count,
    normalize_summary_api,
)
from .services.analysis_service import analyze_video_request
from .services.tf_quiz_service import TF_QUIZ_TYPE, generate_tf_quiz_items, save_tf_questions
from .services.objective_quiz_service import (
    OBJECTIVE_QUIZ_TYPE,
    FEEDBACK_QUIZ_TYPE,
    generate_objective_quiz,
    save_objective_questions_from_text,
    build_objective_items_for_display,
    get_objective_quizzes_for_lecture,
)
from .services.feedback_service import (
    get_wrong_objective_items_from_quiz,
    build_feedback_quiz_text_from_wrong_items,
)
from .services.subjective_quiz_service import (
    generate_quiz,
    save_quiz_questions_from_text,
    build_quiz_items_for_display,
    save_user_answers_to_quiz_answer,
    split_quiz_reference,
)

# 페이지네이션/관리자 화면 설정
HISTORY_PAGE_SIZE = 5
MANAGE_ANSWER_PAGE_SIZE = 10
MANAGE_USER_PAGE_SIZE = 10
HUMAN_LABEL_CHOICES = ["", "정답", "부분 정답", "오답"]
PREDICTED_LABEL_CHOICES = ["", "정답", "부분 정답", "오답"]
_REVIEWED_Q = Q(human_label__isnull=False) & ~Q(human_label="")
USER_ROLE_CHOICES = [
    ("active", _lazy("일반 사용자")),
    ("staff", _lazy("운영자")),
    ("inactive", _lazy("비활성화")),
]

# 업로드 영상 파일 제한
ALLOWED_LECTURE_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
}
MAX_LECTURE_VIDEO_BYTES = 500 * 1024 * 1024

LECTURE_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


def _compute_study_streaks(activity_dates):
    """활동한 날짜 집합에서 (현재 연속일, 최고 연속일)을 계산한다."""
    if not activity_dates:
        return 0, 0

    sorted_dates = sorted(activity_dates)

    longest = 1
    run = 1
    for prev_date, cur_date in zip(sorted_dates, sorted_dates[1:]):
        gap = (cur_date - prev_date).days
        if gap == 1:
            run += 1
        else:
            run = 1
        longest = max(longest, run)

    today = timezone.localdate()
    date_set = set(sorted_dates)

    current = 0
    if today in date_set:
        cursor = today
    elif (today - timedelta(days=1)) in date_set:
        cursor = today - timedelta(days=1)
    else:
        cursor = None

    while cursor is not None and cursor in date_set:
        current += 1
        cursor -= timedelta(days=1)

    return current, longest


def _supported_language_codes():
    return {code for code, _label in settings.LANGUAGES}


def _set_language_cookie(response, lang_code):
    response.set_cookie(
        settings.LANGUAGE_COOKIE_NAME,
        lang_code,
        max_age=settings.LANGUAGE_COOKIE_AGE,
        path=settings.LANGUAGE_COOKIE_PATH,
        domain=settings.LANGUAGE_COOKIE_DOMAIN,
        secure=settings.LANGUAGE_COOKIE_SECURE,
        httponly=settings.LANGUAGE_COOKIE_HTTPONLY,
        samesite=settings.LANGUAGE_COOKIE_SAMESITE,
    )


def _activate_user_language(request, response, lang_code):
    if lang_code not in _supported_language_codes() or not check_for_language(lang_code):
        return
    translation.activate(lang_code)
    request.LANGUAGE_CODE = lang_code
    _set_language_cookie(response, lang_code)


@require_POST
def set_preferred_language(request):
    """사이드바 언어 선택: 쿠키 저장 + 로그인 시 UserProfile.preferred_language 반영."""
    next_url = request.POST.get("next") or request.META.get("HTTP_REFERER") or "/"
    if not url_has_allowed_host_and_scheme(
        url=next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        next_url = "/"

    lang_code = (request.POST.get("language") or "").strip()
    response = HttpResponseRedirect(next_url)

    if lang_code in _supported_language_codes() and check_for_language(lang_code):
        _activate_user_language(request, response, lang_code)
        if request.user.is_authenticated:
            profile, _created = UserProfile.objects.get_or_create(user=request.user)
            if profile.preferred_language != lang_code:
                profile.preferred_language = lang_code
                profile.save(update_fields=["preferred_language"])

    return response


def _build_activity_by_date(user):
    """날짜별 업로드·퀴즈 풀이 수를 집계한다."""
    activity_by_date = {}

    def bump(key, field):
        if key not in activity_by_date:
            activity_by_date[key] = {"uploads": 0, "quizzes": 0}
        activity_by_date[key][field] += 1

    for created_at in Lecture.objects.filter(user=user).values_list("created_at", flat=True):
        if created_at:
            bump(timezone.localtime(created_at).date().isoformat(), "uploads")

    for created_at in QuizAnswer.objects.filter(user=user).values_list("created_at", flat=True):
        if created_at:
            bump(timezone.localtime(created_at).date().isoformat(), "quizzes")

    return activity_by_date


def _decorate_lecture_for_display(lecture):
    """강의 제목·과목 표시용 필드를 lecture 객체에 붙인다."""
    lecture.display_title = lecture.title
    lecture.detected_subject = "4"

    match = re.search(
        r"\[\s*SUB\s*:\s*(auto|[1-4])\s*\]",
        lecture.title or "",
        re.IGNORECASE,
    )

    if match:
        lecture.detected_subject = match.group(1)
        lecture.display_title = re.sub(
            r"\s*\[\s*SUB\s*:\s*(?:auto|[1-4])\s*\]",
            "",
            lecture.title or "",
            flags=re.IGNORECASE,
        ).strip()
    else:
        title_lower = (lecture.title or "").lower()

        if any(k in title_lower for k in ["역사", "인문", "조선", "세계사", "한국사"]):
            lecture.detected_subject = "1"
        elif any(k in title_lower for k in ["코딩", "파이썬", "개발", "프로그래밍", "알고리즘"]):
            lecture.detected_subject = "2"
        elif any(k in title_lower for k in ["수학", "과학", "물리", "화학", "생물", "미적분"]):
            lecture.detected_subject = "3"

    return lecture


def _lecture_card_thumbnail(lecture):
    """홈·히스토리 카드용 썸네일 URL."""
    if lecture.thumbnail_url:
        return lecture.thumbnail_url
    if lecture.video_id:
        return f"https://img.youtube.com/vi/{lecture.video_id}/0.jpg"
    return ""

def home(request):
    """메인(홈) 페이지. 로그인 시 학습 현황 통계를 함께 보여준다."""
    if not request.user.is_authenticated:
        return render(request, "home.html")

    user = request.user
    user_lectures = Lecture.objects.filter(user=user)
    uploaded_count = user_lectures.count()

    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    lectures_this_month = user_lectures.filter(created_at__gte=month_start).count()

    answers = QuizAnswer.objects.filter(user=user)
    solved_count = answers.count()

    correct_count = answers.filter(
        Q(human_label="정답")
        | (Q(human_label="") & Q(predicted_label="정답 가능성 높음"))
    ).count()
    accuracy_rate = round(correct_count / solved_count * 100) if solved_count else 0

    activity_dates = set()
    for created_at in user_lectures.values_list("created_at", flat=True):
        if created_at:
            activity_dates.add(timezone.localtime(created_at).date())
    for created_at in answers.values_list("created_at", flat=True):
        if created_at:
            activity_dates.add(timezone.localtime(created_at).date())

    activity_by_date = _build_activity_by_date(user)
    study_streak, best_streak = _compute_study_streaks(activity_dates)

    latest_answer_subq = QuizAnswer.objects.filter(
        user=user,
        quiz_question__quiz__lecture=OuterRef("pk"),
    ).order_by("-created_at").values("created_at")[:1]

    recent_lectures = list(
        user_lectures.annotate(
            quiz_count=Count(
                "quizzes",
                filter=Q(quizzes__quiz_text__gt=""),
                distinct=True,
            ),
            last_answer_at=Subquery(latest_answer_subq),
            last_activity=Coalesce("last_answer_at", "created_at"),
        )
        .order_by("-last_activity", "-id")[:3]
    )
    for lecture in recent_lectures:
        _decorate_lecture_for_display(lecture)
        lecture.card_thumbnail = _lecture_card_thumbnail(lecture)

    calendar_memos = {
        memo.date.isoformat(): memo.memo
        for memo in StudyCalendarMemo.objects.filter(user=user).only("date", "memo")
        if memo.memo.strip()
    }

    context = {
        "uploaded_count": uploaded_count,
        "lectures_this_month": lectures_this_month,
        "solved_count": solved_count,
        "accuracy_rate": accuracy_rate,
        "study_streak": study_streak,
        "best_streak": best_streak,
        "activity_dates": sorted(d.isoformat() for d in activity_dates),
        "activity_by_date": activity_by_date,
        "calendar_memos": calendar_memos,
        "recent_lectures": recent_lectures,
    }
    return render(request, "home.html", context)


@login_required(login_url="home")
def save_calendar_memo(request):
    """캘린더 날짜별 학습 메모 저장·삭제."""
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": _("POST만 허용됩니다.")}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"ok": False, "error": _("잘못된 요청입니다.")}, status=400)

    date_str = (payload.get("date") or "").strip()
    memo = (payload.get("memo") or "").strip()

    try:
        memo_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return JsonResponse({"ok": False, "error": _("날짜 형식이 올바르지 않습니다.")}, status=400)

    if len(memo) > 500:
        return JsonResponse({"ok": False, "error": _("메모는 500자까지 입력할 수 있습니다.")}, status=400)

    if not memo:
        StudyCalendarMemo.objects.filter(user=request.user, date=memo_date).delete()
        return JsonResponse({"ok": True, "deleted": True, "date": date_str})

    entry, _created = StudyCalendarMemo.objects.update_or_create(
        user=request.user,
        date=memo_date,
        defaults={"memo": memo},
    )
    return JsonResponse({
        "ok": True,
        "date": date_str,
        "memo": entry.memo,
        "updated_at": timezone.localtime(entry.updated_at).strftime("%Y-%m-%d %H:%M"),
    })


def login_page(request):
    """로그인 처리 엔드포인트."""
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)

        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, _("%(username)s님, 환영합니다!") % {"username": user.username})
            response = redirect("home")
            try:
                lang_code = user.profile.preferred_language
            except UserProfile.DoesNotExist:
                lang_code = settings.LANGUAGE_CODE
            _activate_user_language(request, response, lang_code)
            return response

        messages.error(request, _("아이디 또는 비밀번호가 올바르지 않습니다."))
        return redirect("home")

    return redirect("home")


def logout_page(request):
    """로그아웃 후 홈으로 이동."""
    logout(request)
    return redirect("home")

def signup_page(request):
    """회원가입 및 추가 프로필 정보 저장."""
    if request.method == "POST":
        form = UserCreationForm(request.POST)

        if form.is_valid():
            user = form.save()

            user_name = request.POST.get("name", "")
            user_age = request.POST.get("age", "")
            user_phone = request.POST.get("phone", "")
            cert_name = request.POST.get("certification", "")
            reason = request.POST.get("reason", "")
            interest = request.POST.get("interest", "")

            UserProfile.objects.create(
                user=user,
                name=user_name,
                age=int(user_age) if user_age else None,
                phone=user_phone,
                certification=cert_name,
                reason=reason,
                interest=interest,
            )

            messages.success(request, "회원가입이 완료되었습니다. 로그인해주세요.")
            return redirect("home")

        messages.error(request, "회원가입에 실패했습니다. 입력값을 다시 확인해주세요.")
        return render(request, "signup.html", {"form": form})

    form = UserCreationForm()
    return render(request, "signup.html", {"form": form})

def validate_lecture_video_file(uploaded_file):
    """업로드 영상 파일 확장자·용량 검증. 문제 없으면 None."""
    if not uploaded_file:
        return "분석할 영상 파일을 선택해 주세요."

    ext = os.path.splitext(uploaded_file.name)[1].lower()

    if ext not in ALLOWED_LECTURE_VIDEO_EXTENSIONS:
        return f"지원하지 않는 영상 형식입니다. ({ext or '확장자 없음'})"

    if uploaded_file.size > MAX_LECTURE_VIDEO_BYTES:
        return "영상 파일은 500MB 이하여야 합니다."

    return None

@login_required(login_url="home")
def upload_page(request):
    """강의 업로드 페이지."""
    if request.method == "POST":
        title = (request.POST.get("title") or "").strip()
        upload_mode = (request.POST.get("upload_mode") or Lecture.SOURCE_YOUTUBE).strip()

        whisper_model = normalize_whisper_model_name(
            request.POST.get("whisper_model", "base")
        )
        workers = normalize_worker_count(
            request.POST.get("workers", 2)
        )
        subject_code = normalize_subject_code(
            request.POST.get("subject_code", "auto")
        )
        summary_api = normalize_summary_api(
            request.POST.get("summary_api", "gemini")
        )

        clean_title = re.sub(
            r"\s*\[\s*SUB\s*:\s*(?:auto|[1-4])\s*\]",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()

        if not clean_title:
            clean_title = "제목 없는 강의"

        final_title = f"{clean_title} [SUB:{subject_code}]"

        if upload_mode == Lecture.SOURCE_FILE:
            lecture_file = request.FILES.get("lecture_file")
            file_error = validate_lecture_video_file(lecture_file)

            if file_error:
                messages.error(request, file_error)
                return redirect("upload")

            lecture = Lecture.objects.create(
                user=request.user,
                title=final_title,
                source_type=Lecture.SOURCE_FILE,
                youtube_url="",
                video_file=lecture_file,
                video_id="",
                thumbnail_url="",
            )

            thumbnail_url = capture_lecture_video_thumbnail(lecture)

            if thumbnail_url:
                lecture.thumbnail_url = thumbnail_url
                lecture.save(update_fields=["thumbnail_url"])

        else:
            youtube_link = (request.POST.get("youtube_link") or "").strip()

            if not youtube_link:
                messages.error(request, "유튜브 링크를 입력해 주세요.")
                return redirect("upload")

            video_id = extract_youtube_video_id(youtube_link)
            thumbnail_url = ""

            if video_id:
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"

            lecture = Lecture.objects.create(
                user=request.user,
                title=final_title,
                source_type=Lecture.SOURCE_YOUTUBE,
                youtube_url=youtube_link,
                video_id=video_id or "",
                thumbnail_url=thumbnail_url,
            )

        request.session[f"lecture_analysis_options_{lecture.id}"] = {
            "whisper_model": whisper_model,
            "workers": workers,
            "subject_code": subject_code,
            "summary_api": summary_api,
        }
        request.session.modified = True

        print(
            f"[분석 옵션 저장] lecture_id={lecture.id}, "
            f"source_type={lecture.source_type}, "
            f"whisper_model={whisper_model}, "
            f"workers={workers}, "
            f"subject_code={subject_code}, "
            f"summary_api={summary_api}"
        )

        return redirect(f"/summary/?lecture_id={lecture.id}")

    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    user_lectures = Lecture.objects.filter(user=request.user)

    lectures_this_month = user_lectures.filter(
        created_at__gte=month_start,
    ).exclude(summary_text="").count()

    recent_lecture_ids = list(
        user_lectures.exclude(summary_text="")
        .order_by("-created_at")[:5]
        .values_list("id", flat=True)
    )

    avg_result = (
        QuizAnswer.objects.filter(
            user=request.user,
            quiz_question__quiz__lecture_id__in=recent_lecture_ids,
            similarity_score__isnull=False,
        ).aggregate(avg=Avg("similarity_score"))
    )
    average_similarity = round((avg_result["avg"] or 0) * 100, 1)

    total_quiz_count = QuizQuestion.objects.filter(
        quiz__lecture__user=request.user,
    ).count()

    return render(
        request,
        "upload.html",
        {
            "lectures_this_month": lectures_this_month,
            "average_similarity": average_similarity,
            "total_quiz_count": total_quiz_count,
        },
    )


@login_required(login_url="home")
def summary_page(request):
    """요약 페이지 렌더링."""
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return redirect("upload")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    video_id = ""

    if lecture.source_type == Lecture.SOURCE_YOUTUBE:
        video_id = (lecture.video_id or "").strip() or extract_youtube_video_id(
            lecture.youtube_url or ""
        ) or ""

    video_file_name = ""
    video_file_url = ""

    if lecture.video_file:
        video_file_name = os.path.basename(lecture.video_file.name)

        if lecture.source_type == Lecture.SOURCE_FILE:
            try:
                video_file_url = reverse("stream_lecture_video", args=[lecture.id])
            except NoReverseMatch:
                video_file_url = lecture.video_file.url
        else:
            video_file_url = lecture.video_file.url

    context = {
        "lecture_id": lecture.id,
        "lecture_title": lecture.title,
        "source_type": lecture.source_type,
        "youtube_link": lecture.youtube_url,
        "video_id": video_id,
        "thumbnail_url": lecture.thumbnail_url,
        "video_file_name": video_file_name,
        "video_file_url": video_file_url,
    }

    return render(request, "summary.html", context)


@login_required(login_url="home")
def download_summary_pdf(request, lecture_id):
    """강의 요약 PDF 다운로드."""
    from .summary_pdf import build_summary_pdf_filename, generate_summary_pdf_bytes

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if not (lecture.summary_text or "").strip():
        return HttpResponse(
            "요약이 아직 없습니다.",
            status=404,
            content_type="text/plain; charset=utf-8",
        )

    try:
        pdf_bytes = generate_summary_pdf_bytes(lecture)
    except Exception as exc:
        print(f"[PDF 다운로드 오류] lecture_id={lecture_id}: {exc}")
        return HttpResponse(
            "PDF 생성에 실패했습니다. static/fonts/NotoSansKR-Regular.ttf 파일을 확인해 주세요.",
            status=500,
            content_type="text/plain; charset=utf-8",
        )

    filename = build_summary_pdf_filename(lecture)
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response

@login_required(login_url="home")
def stream_lecture_video(request, lecture_id):
    """업로드 영상을 HTTP Range(206)로 스트리밍해 재생·탐색이 즉시 가능하도록 한다."""
    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if lecture.source_type != Lecture.SOURCE_FILE or not lecture.video_file:
        raise Http404("영상 파일이 없습니다.")

    video_path = lecture.video_file.path

    if not os.path.exists(video_path):
        raise Http404("영상 파일을 찾을 수 없습니다.")

    ext = os.path.splitext(video_path)[1].lower()
    content_type = LECTURE_VIDEO_MIME_TYPES.get(ext, "application/octet-stream")
    file_size = os.path.getsize(video_path)
    range_header = request.META.get("HTTP_RANGE", "").strip()

    if range_header:
        range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)

        if range_match:
            start = int(range_match.group(1))
            end_str = range_match.group(2)
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)

            if start >= file_size or start > end:
                response = HttpResponse(status=416)
                response["Content-Range"] = f"bytes */{file_size}"
                return response

            length = end - start + 1

            with open(video_path, "rb") as video_fp:
                video_fp.seek(start)
                chunk = video_fp.read(length)

            response = HttpResponse(chunk, status=206, content_type=content_type)
            response["Content-Length"] = str(length)
            response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            response["Accept-Ranges"] = "bytes"
            return response

    response = FileResponse(open(video_path, "rb"), content_type=content_type)
    response["Content-Length"] = str(file_size)
    response["Accept-Ranges"] = "bytes"
    return response

@login_required(login_url="home")
def analyze_video_api(request):
    """강의 분석 API wrapper. 실제 분석 흐름은 services/analysis_service.py에서 처리한다."""
    return analyze_video_request(request)

@login_required(login_url="home")
def history_page(request):
    """사용자별 강의 히스토리 페이지."""

    search_query = (request.GET.get("search") or "").strip()
    quiz_filter = (request.GET.get("quiz") or "all").strip()
    duration_filter = (request.GET.get("duration") or "all").strip()
    subject_filter = (request.GET.get("subject") or "all").strip()
    sort_by = (request.GET.get("sort") or "-created_at").strip()

    allowed_sort_values = ["-created_at", "created_at", "title"]

    if sort_by not in allowed_sort_values:
        sort_by = "-created_at"

    quiz_exists = Quiz.objects.filter(
        lecture=OuterRef("pk")
    ).exclude(quiz_text="")

    lectures = (
        Lecture.objects
        .filter(user=request.user)
        .annotate(has_quiz=Exists(quiz_exists))
        .annotate(
            quiz_count=Count(
                "quizzes",
                filter=Q(quizzes__quiz_text__gt=""),
                distinct=True,
            ),
            latest_generation=Max("quizzes__generation_number"),
        )
    )

    if search_query:
        lectures = lectures.filter(title__icontains=search_query)

    if quiz_filter == "quiz_exist":
        lectures = lectures.filter(has_quiz=True)
    elif quiz_filter == "no_quiz":
        lectures = lectures.filter(has_quiz=False)
    else:
        quiz_filter = "all"

    if duration_filter == "short":
        lectures = lectures.filter(analysis_duration_seconds__lt=600)
    elif duration_filter == "medium":
        lectures = lectures.filter(
            analysis_duration_seconds__gte=600,
            analysis_duration_seconds__lte=1800,
        )
    elif duration_filter == "long":
        lectures = lectures.filter(analysis_duration_seconds__gt=1800)
    else:
        duration_filter = "all"

    if subject_filter in ["auto", "1", "2", "3", "4"]:
        lectures = lectures.filter(title__icontains=f"[SUB:{subject_filter}]")
    else:
        subject_filter = "all"

    lectures = lectures.order_by(sort_by, "-id")

    paginator = Paginator(lectures, HISTORY_PAGE_SIZE)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    for lecture in page_obj:
        _decorate_lecture_for_display(lecture)

        if (
            lecture.source_type == Lecture.SOURCE_FILE
            and not lecture.thumbnail_url
            and lecture.video_file
        ):
            thumbnail_url = capture_lecture_video_thumbnail(lecture)

            if thumbnail_url:
                lecture.thumbnail_url = thumbnail_url
                lecture.save(update_fields=["thumbnail_url"])

    params = request.GET.copy()

    if "page" in params:
        del params["page"]

    base_query = params.urlencode()

    return render(
        request,
        "history.html",
        {
            "page_obj": page_obj,
            "lectures": page_obj,
            "search_query": search_query,
            "quiz_filter": quiz_filter,
            "duration_filter": duration_filter,
            "subject_filter": subject_filter,
            "sort_by": sort_by,
            "base_query": base_query,
        },
    )


@login_required(login_url="home")
def quiz_page(request):
    """객관식 문제 생성, 표시, 사용자 선택 답안 저장 페이지."""
    lecture_id = request.GET.get("lecture_id")

    if request.method == "POST":
        lecture_id = lecture_id or request.POST.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if not lecture.summary_text:
        return redirect(f"/summary/?lecture_id={lecture.id}")

    quizzes = get_objective_quizzes_for_lecture(lecture).order_by("generation_number")

    if request.method == "POST" and request.POST.get("generate_quiz"):
        max_generation = quizzes.aggregate(max_number=Max("generation_number"))["max_number"] or 0
        next_generation = max_generation + 1
        previous_quiz_texts = "\n\n".join([quiz.quiz_text for quiz in quizzes if quiz.quiz_text])

        quiz_text = generate_objective_quiz(
            summary_text=lecture.summary_text,
            question_count=20,
            previous_quiz_texts=previous_quiz_texts,
        )

        new_quiz = Quiz.objects.create(
            lecture=lecture,
            generation_number=next_generation,
            question="",
            answer="",
            explanation=OBJECTIVE_QUIZ_TYPE,
            quiz_text=quiz_text,
        )
        save_objective_questions_from_text(new_quiz, quiz_text)

        return redirect(f"/quiz/?lecture_id={lecture.id}&generation={next_generation}")

    if not quizzes.exists():
        quiz_text = generate_objective_quiz(
            summary_text=lecture.summary_text,
            question_count=20,
            previous_quiz_texts="",
        )
        first_quiz = Quiz.objects.create(
            lecture=lecture,
            generation_number=1,
            question="",
            answer="",
            explanation=OBJECTIVE_QUIZ_TYPE,
            quiz_text=quiz_text,
        )
        save_objective_questions_from_text(first_quiz, quiz_text)
        return redirect(f"/quiz/?lecture_id={lecture.id}&generation=1")

    selected_generation = request.GET.get("generation") or request.POST.get("generation")
    selected_quiz = None

    if selected_generation and str(selected_generation).isdigit():
        selected_quiz = quizzes.filter(generation_number=int(selected_generation)).first()

    if not selected_quiz:
        selected_quiz = quizzes.order_by("-generation_number", "-id").first()

    if not selected_quiz:
        selected_quiz = quizzes.first()

    if request.method == "POST" and request.POST.get("save_user_answers"):
        for key, value in request.POST.items():
            if not key.startswith("question_"):
                continue

            question_id = key.replace("question_", "", 1)

            if not str(question_id).isdigit():
                continue

            try:
                question = QuizQuestion.objects.get(
                    id=int(question_id),
                    quiz=selected_quiz,
                )
            except QuizQuestion.DoesNotExist:
                continue

            selected_choice = str(value or "").strip()
            selected_choice = selected_choice.replace("①", "1").replace("②", "2").replace("③", "3").replace("④", "4")

            match = re.search(r"[1-4]", selected_choice)
            if match:
                selected_choice = match.group(0)

            QuizAnswer.objects.update_or_create(
                user=request.user,
                quiz_question=question,
                defaults={
                    "user_answer": selected_choice,
                    "similarity_score": None,
                    "predicted_label": "",
                },
            )

        return redirect(f"/quiz/answer/?lecture_id={lecture.id}&generation={selected_quiz.generation_number}")

    quiz_items = build_objective_items_for_display(selected_quiz, request.user)

    all_generations = list(quizzes.values_list("generation_number", flat=True).order_by("generation_number"))
    current_gen = selected_quiz.generation_number if selected_quiz else 1

    current_idx = all_generations.index(current_gen) if current_gen in all_generations else 0
    prev_gen = all_generations[current_idx - 1] if current_idx > 0 else None
    next_gen = all_generations[current_idx + 1] if current_idx < len(all_generations) - 1 else None

    return render(request, "quiz.html", {
        "lecture": lecture,
        "quizzes": quizzes,
        "selected_quiz": selected_quiz,
        "quiz_items": quiz_items,
        "current_gen": current_gen,
        "all_generations": all_generations,
        "has_prev": prev_gen is not None,
        "has_next": next_gen is not None,
        "prev_gen": prev_gen,
        "next_gen": next_gen,
    })


@login_required(login_url="home")
def quiz_answer_page(request):
    """객관식 정답 및 오답 확인 페이지."""
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if not lecture.summary_text:
        return redirect(f"/summary/?lecture_id={lecture.id}")

    quizzes = get_objective_quizzes_for_lecture(lecture)

    if not quizzes.exists():
        return redirect(f"/quiz/?lecture_id={lecture.id}")

    selected_generation = request.GET.get("generation")
    selected_quiz = None

    if selected_generation and str(selected_generation).isdigit():
        selected_quiz = quizzes.filter(generation_number=int(selected_generation)).first()

    if not selected_quiz:
        selected_quiz = quizzes.order_by("-generation_number", "-id").first()

    if not selected_quiz:
        selected_quiz = quizzes.first()

    quiz_items = build_objective_items_for_display(selected_quiz, request.user)

    correct_count = 0
    wrong_count = 0
    has_answers = False
    wrong_items = []

    for item in quiz_items:
        user_choice = str(item.get("user_choice") or "").strip()
        correct_choice = str(item.get("correct_choice") or "").strip()
        item["is_correct"] = bool(user_choice) and user_choice == correct_choice

        selected_text = ""
        correct_text = ""

        for choice in item.get("choices") or []:
            c_num = str(choice.get("c_num") or "").strip()
            choice_label = f"{c_num}번"
            choice_text = choice.get("choice_text") or ""

            if c_num == user_choice:
                selected_text = f"{choice_label} {choice_text}"

            if c_num == correct_choice:
                correct_text = f"{choice_label} {choice_text}"

        item["selected_text"] = selected_text
        item["correct_text"] = correct_text

        if user_choice:
            has_answers = True

        if item["is_correct"]:
            correct_count += 1
        else:
            wrong_count += 1
            wrong_items.append(item)

    total_count = len(quiz_items)
    score = round((correct_count / total_count) * 100, 1) if total_count else 0

    # 피드백 페이지에서 오답 기반 서술형 문제 생성에 활용할 수 있도록 세션에도 저장한다.
    request.session[f"objective_wrong_items_{lecture.id}"] = [
        {
            "objective_number": item.get("q_num") or item.get("number"),
            "question_number": item.get("q_num") or item.get("number"),
            "question_text": item.get("question_text"),
            "keyword": item.get("keyword"),
            "timeline": item.get("timeline"),
            "explanation": item.get("explanation"),
            "correct_text": item.get("correct_text"),
            "selected_text": item.get("selected_text"),
            "user_choice": item.get("user_choice"),
            "correct_choice": item.get("correct_choice"),
            "choices": item.get("choices"),
        }
        for item in wrong_items
    ]
    request.session.modified = True

    return render(request, "quiz_answer.html", {
        "lecture": lecture,
        "quizzes": quizzes,
        "selected_quiz": selected_quiz,
        "result_items": quiz_items,
        "quiz_items": quiz_items,
        "total_count": total_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "score": score,
        "has_answers": has_answers,
        "wrong_items": wrong_items,
    })


@login_required(login_url="home")
def tf_quiz_page(request):
    """O/X 퀴즈 생성, 표시, 사용자 선택 답안 저장 페이지."""
    lecture_id = request.GET.get("lecture_id")

    if request.method == "POST":
        lecture_id = lecture_id or request.POST.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if not lecture.summary_text:
        return redirect(f"/summary/?lecture_id={lecture.id}")

    # [핵심] explanation 필드를 "tf" 값으로 지정하여 O/X 퀴즈만 조회
    quizzes = Quiz.objects.filter(lecture=lecture, explanation=TF_QUIZ_TYPE).order_by("generation_number")

    # 1. 새 O/X 퀴즈 생성 요청 처리
    if request.method == "POST" and request.POST.get("generate_quiz"):
        max_generation = quizzes.aggregate(max_number=Max("generation_number"))["max_number"] or 0
        next_generation = max_generation + 1

        new_quiz = Quiz.objects.create(
            lecture=lecture,
            generation_number=next_generation,
            question="",
            answer="",
            explanation=TF_QUIZ_TYPE,  # explanation="tf"로 지정
            quiz_text="",
        )
        items = generate_tf_quiz_items(lecture.summary_text, question_count=20)
        save_tf_questions(new_quiz, items)

        return redirect(f"/tf-quiz/?lecture_id={lecture.id}&generation={next_generation}")

    # 2. 최초 진입 시 O/X 퀴즈가 없는 경우 자동 생성
    if not quizzes.exists():
        first_quiz = Quiz.objects.create(
            lecture=lecture,
            generation_number=1,
            question="",
            answer="",
            explanation=TF_QUIZ_TYPE,  # explanation="tf"로 지정
            quiz_text="",
        )
        items = generate_tf_quiz_items(lecture.summary_text, question_count=20)
        save_tf_questions(first_quiz, items)
        return redirect(f"/tf-quiz/?lecture_id={lecture.id}&generation=1")

    # 3. 요청된 회차(generation) 선택
    selected_generation = request.GET.get("generation") or request.GET.get("gen")
    selected_quiz = None

    if selected_generation and str(selected_generation).isdigit():
        selected_quiz = quizzes.filter(generation_number=int(selected_generation)).first()

    if not selected_quiz:
        selected_quiz = quizzes.order_by("-generation_number", "-id").first()

    if not selected_quiz:
        selected_quiz = quizzes.first()

    show_result = False

    # 4. O/X 답안 채점 및 저장
    if request.method == "POST" and request.POST.get("save_user_answers"):
        for key, value in request.POST.items():
            if not key.startswith("question_"):
                continue

            question_id = key.replace("question_", "", 1)
            if not str(question_id).isdigit():
                continue

            try:
                question = QuizQuestion.objects.get(
                    id=int(question_id),
                    quiz=selected_quiz,
                )
            except QuizQuestion.DoesNotExist:
                continue

            user_choice = str(value or "").strip().upper()  # 'O' 또는 'X'

            QuizAnswer.objects.update_or_create(
                user=request.user,
                quiz_question=question,
                defaults={
                    "user_answer": user_choice,
                    "similarity_score": None,
                    "predicted_label": "",
                },
            )
        show_result = True

    # 5. 화면 표출용 아이템 구성
    quiz_questions = selected_quiz.questions.all().order_by("number")
    quiz_items = []

    for q in quiz_questions:
        user_ans = ""
        user_answer_obj = QuizAnswer.objects.filter(user=request.user, quiz_question=q).first()
        if user_answer_obj:
            user_ans = user_answer_obj.user_answer

        # explanation JSON 파싱 (해설 및 원문)
        explanation_text = ""
        original_sentence = ""
        if q.explanation:
            try:
                meta = json.loads(q.explanation)
                explanation_text = meta.get("explanation", "")
                original_sentence = meta.get("original_sentence", "")
            except Exception:
                explanation_text = q.explanation

        quiz_items.append({
            "id": q.id,
            "number": q.number,
            "question_text": q.question_text,
            "model_answer": q.model_answer,
            "user_answer": user_ans,
            "is_correct": (user_ans == q.model_answer) if user_ans else False,
            "explanation": explanation_text,
            "original_sentence": original_sentence,
        })

    # 6. 회차 이동 네비게이션용 계산
    all_generations = list(quizzes.values_list("generation_number", flat=True).order_by("generation_number"))
    current_gen = selected_quiz.generation_number if selected_quiz else 1

    prev_gen = current_gen - 1 if (current_gen - 1) in all_generations else None
    next_gen = current_gen + 1 if (current_gen + 1) in all_generations else None

    return render(request, "tf_quiz.html", {
        "lecture": lecture,
        "quizzes": quizzes,
        "selected_quiz": selected_quiz,
        "quiz_items": quiz_items,
        "show_result": show_result,
        "current_gen": current_gen,
        "has_prev": prev_gen is not None,
        "has_next": next_gen is not None,
        "prev_gen": prev_gen,
        "next_gen": next_gen,
    })


@login_required(login_url="home")
def feedback_page(request):
    """학습 피드백 페이지.

    객관식 오답 이후 제공되는 서술형 피드백 문제 생성,
    사용자 서술형 답안 저장, Reranker 기반 자동 채점 결과 확인을 담당한다.
    """
    lecture_id = request.GET.get("lecture_id")

    if request.method == "POST":
        lecture_id = lecture_id or request.POST.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)
    has_summary = bool(lecture.summary_text)

    if not has_summary:
        return render(request, "feedback.html", {
            "lecture": lecture,
            "has_summary": False,
            "has_quiz": False,
            "quizzes": [],
            "selected_quiz": None,
            "quiz_items": [],
            "answer_items": [],
            "total_answer_count": 0,
            "selected_generation": None,
            "average_similarity": 0,
            "correct_like_count": 0,
            "review_needed_count": 0,
            "wrong_like_count": 0,
            "reviewed_count": 0,
            "review_agree_count": 0,
            "review_agreement_rate": None,
            "objective_generation": None,
        })

    objective_generation = request.GET.get("objective_generation") or request.POST.get("objective_generation")
    auto_generate_feedback = request.GET.get("auto_generate_feedback") == "1"

    feedback_quizzes = (
        Quiz.objects
        .filter(lecture=lecture, explanation=FEEDBACK_QUIZ_TYPE)
        .exclude(quiz_text="")
        .order_by("generation_number", "id")
    )

    def create_feedback_quiz_from_objective(next_generation):
        wrong_items = []

        if objective_generation:
            objective_quizzes = get_objective_quizzes_for_lecture(lecture)
            objective_quiz = None

            if str(objective_generation).isdigit():
                objective_quiz = objective_quizzes.filter(generation_number=int(objective_generation)).first()

            if objective_quiz:
                wrong_items = get_wrong_objective_items_from_quiz(request.user, objective_quiz)

        if not wrong_items:
            wrong_items = request.session.get(f"objective_wrong_items_{lecture.id}", []) or []

        return build_feedback_quiz_text_from_wrong_items(
            wrong_items=wrong_items,
            generation_number=next_generation,
            summary_text=lecture.summary_text,
        )

    if auto_generate_feedback and objective_generation:
        max_generation = feedback_quizzes.aggregate(max_number=Max("generation_number"))["max_number"] or 0
        next_generation = max_generation + 1
        quiz_text = create_feedback_quiz_from_objective(next_generation)

        new_quiz = Quiz.objects.create(
            lecture=lecture,
            generation_number=next_generation,
            question=quiz_text,
            answer="",
            explanation=FEEDBACK_QUIZ_TYPE,
            quiz_text=quiz_text,
        )
        save_quiz_questions_from_text(new_quiz, quiz_text)

        return redirect(
            f"/feedback/?lecture_id={lecture.id}"
            f"&generation={next_generation}"
            f"&objective_generation={objective_generation}"
        )

    if request.method == "POST" and request.POST.get("generate_feedback_questions"):
        max_generation = feedback_quizzes.aggregate(max_number=Max("generation_number"))["max_number"] or 0
        next_generation = max_generation + 1

        if objective_generation:
            quiz_text = create_feedback_quiz_from_objective(next_generation)
        else:
            previous_quiz_texts = "\n\n".join([quiz.quiz_text for quiz in feedback_quizzes])
            quiz_text = generate_quiz(
                lecture.summary_text,
                generation_number=next_generation,
                previous_quiz_texts=previous_quiz_texts,
            )

        new_quiz = Quiz.objects.create(
            lecture=lecture,
            generation_number=next_generation,
            question=quiz_text,
            answer="",
            explanation=FEEDBACK_QUIZ_TYPE,
            quiz_text=quiz_text,
        )
        save_quiz_questions_from_text(new_quiz, quiz_text)

        redirect_url = f"/feedback/?lecture_id={lecture.id}&generation={next_generation}"
        if objective_generation:
            redirect_url += f"&objective_generation={objective_generation}"
        return redirect(redirect_url)

    if request.method == "POST" and request.POST.get("save_feedback_answers"):
        gen_raw = request.POST.get("generation")

        if not gen_raw or not str(gen_raw).isdigit():
            return redirect(f"/feedback/?lecture_id={lecture.id}")

        gen = int(gen_raw)
        quiz_to_update = feedback_quizzes.filter(generation_number=gen).first()

        if not quiz_to_update:
            return redirect(f"/feedback/?lecture_id={lecture.id}")

        save_user_answers_to_quiz_answer(
            user=request.user,
            quiz=quiz_to_update,
            post_data=request.POST,
        )

        redirect_url = f"/feedback/?lecture_id={lecture.id}&generation={gen}"
        if objective_generation:
            redirect_url += f"&objective_generation={objective_generation}"
        return redirect(redirect_url)

    feedback_quizzes = (
        Quiz.objects
        .filter(lecture=lecture, explanation=FEEDBACK_QUIZ_TYPE)
        .exclude(quiz_text="")
        .order_by("generation_number", "id")
    )

    has_quiz = feedback_quizzes.exists()
    selected_quiz = None
    selected_generation = None
    quiz_items = []
    raw_user_answer = ""

    if has_quiz:
        generation_param = request.GET.get("generation")

        if generation_param and str(generation_param).isdigit():
            selected_quiz = feedback_quizzes.filter(generation_number=int(generation_param)).first()

        if not selected_quiz:
            selected_quiz = feedback_quizzes.order_by("-generation_number", "-id").first()

        if not selected_quiz:
            selected_quiz = feedback_quizzes.first()

        selected_generation = selected_quiz.generation_number

        quiz_items, raw_user_answer = build_quiz_items_for_display(
            quiz_text=selected_quiz.quiz_text,
            quiz=selected_quiz,
            user=request.user,
        )

    answers = QuizAnswer.objects.none()

    if selected_quiz:
        answers = (
            QuizAnswer.objects
            .filter(user=request.user, quiz_question__quiz=selected_quiz)
            .select_related("quiz_question", "quiz_question__quiz")
            .order_by("quiz_question__number", "created_at")
        )

    answer_items = []

    def normalize_label(label):
        if not label:
            return ""
        label = str(label).strip()
        label_no_space = label.replace(" ", "")
        if label_no_space in ["정답", "정답가능성높음"]:
            return "정답"
        if label_no_space in ["부분정답", "검토필요"]:
            return "부분정답"
        if label_no_space in ["오답", "오답가능성높음"]:
            return "오답"
        return label

    def label_from_human_score(score):
        if score is None:
            return ""
        try:
            score = float(score)
        except (TypeError, ValueError):
            return ""
        if score >= 75:
            return "정답"
        if score >= 45:
            return "부분정답"
        return "오답"

    correct_like_count = 0
    review_needed_count = 0
    wrong_like_count = 0
    reviewed_count = 0
    review_agree_count = 0
    score_sum = 0.0
    total_answer_count = 0

    for answer in answers:
        raw_score = answer.similarity_score if answer.similarity_score is not None else 0.0
        display_score = max(0.0, min(raw_score, 1.0))
        score_sum += display_score
        total_answer_count += 1

        predicted_label = answer.predicted_label or ""
        normalized_predicted = normalize_label(predicted_label)

        if normalized_predicted == "정답":
            correct_like_count += 1
        elif normalized_predicted == "부분정답":
            review_needed_count += 1
        elif normalized_predicted == "오답":
            wrong_like_count += 1

        human_score = answer.human_score
        human_label = answer.human_label or ""
        is_reviewed = bool(human_label) or human_score is not None

        if is_reviewed:
            reviewed_count += 1
            normalized_human_label = normalize_label(human_label)
            if not normalized_human_label:
                normalized_human_label = label_from_human_score(human_score)
            if normalized_predicted == normalized_human_label:
                review_agree_count += 1

        question = answer.quiz_question
        keywords = (question.explanation or "").strip()
        timelines = (getattr(question, "related_timeline", "") or "").strip()

        if not keywords and timelines:
            keywords, timelines = split_quiz_reference(combined=timelines)

        answer_items.append({
            "id": answer.id,
            "generation_number": question.quiz.generation_number,
            "question_number": question.number,
            "question_text": question.question_text,
            "model_answer": question.model_answer,
            "keywords": keywords,
            "timelines": timelines,
            "user_answer": answer.user_answer,
            "similarity_score": raw_score,
            "similarity_percent": round(display_score * 100, 1),
            "predicted_label": predicted_label,
            "human_score": human_score,
            "human_label": human_label,
            "is_reviewed": is_reviewed,
            "created_at": answer.created_at,
        })

    average_similarity = round((score_sum / total_answer_count) * 100, 1) if total_answer_count else 0
    review_agreement_rate = round((review_agree_count / reviewed_count) * 100, 1) if reviewed_count else None

    return render(request, "feedback.html", {
        "lecture": lecture,
        "has_summary": has_summary,
        "has_quiz": has_quiz,
        "quizzes": feedback_quizzes,
        "selected_quiz": selected_quiz,
        "selected_generation": selected_generation,
        "quiz_items": quiz_items,
        "raw_user_answer": raw_user_answer,
        "answer_items": answer_items,
        "total_answer_count": total_answer_count,
        "average_similarity": average_similarity,
        "correct_like_count": correct_like_count,
        "review_needed_count": review_needed_count,
        "wrong_like_count": wrong_like_count,
        "reviewed_count": reviewed_count,
        "review_agree_count": review_agree_count,
        "review_agreement_rate": review_agreement_rate,
        "objective_generation": objective_generation,
    })

def test_api(request):
    """프론트-백엔드 연결 테스트용 API."""
    return JsonResponse(
        {"message": "백엔드 연결 성공"},
        json_dumps_params={"ensure_ascii": False},
    )

def _user_role_value(user):
    if not user.is_active:
        return "inactive"
    if user.is_staff:
        return "staff"
    return "active"

def _apply_user_role(user, role, actor):
    """사용자 권한(is_staff / is_active)을 적용한다."""
    if role == "staff":
        is_staff, is_active = True, True
    elif role == "inactive":
        is_staff, is_active = False, False
    elif role == "active":
        is_staff, is_active = False, True
    else:
        return False, "올바르지 않은 권한 값입니다."

    if user.pk == actor.pk:
        if not is_active:
            return False, "본인 계정은 비활성화할 수 없습니다."
        if not is_staff:
            return False, "본인 계정의 관리자 권한은 해제할 수 없습니다."

    user.is_staff = is_staff
    user.is_active = is_active
    user.save(update_fields=["is_staff", "is_active"])
    return True, ""

def staff_required(view_func):
    """is_staff 사용자만 접근 가능. 비로그인/권한 없음은 홈으로 보낸다."""
    return user_passes_test(
        lambda u: u.is_active and u.is_staff,
        login_url="home",
    )(view_func)

@staff_required
def manage_dashboard(request):
    """학습 관리 대시보드: 핵심 지표와 최근 답안을 한눈에 보여준다."""
    answer_count = QuizAnswer.objects.count()
    reviewed_count = QuizAnswer.objects.filter(_REVIEWED_Q).count()

    recent_answers = (
        QuizAnswer.objects
        .select_related(
            "user",
            "quiz_question",
            "quiz_question__quiz",
            "quiz_question__quiz__lecture",
        )
        .order_by("-created_at")[:8]
    )

    context = {
        "lecture_count": Lecture.objects.count(),
        "analyzed_count": Lecture.objects.exclude(summary_text="").count(),
        "quiz_count": Quiz.objects.exclude(quiz_text="").count(),
        "question_count": QuizQuestion.objects.count(),
        "answer_count": answer_count,
        "reviewed_count": reviewed_count,
        "pending_review_count": max(0, answer_count - reviewed_count),
        "user_count": User.objects.count(),
        "recent_answers": recent_answers,
    }
    return render(request, "manage/dashboard.html", context)

@staff_required
def manage_answers(request):
    """제출된 답안 목록 검수 페이지.

    GET: 필터/검색/페이지네이션으로 답안을 조회한다.
    POST: 특정 답안에 사람 라벨/점수를 저장한다.
    """
    if request.method == "POST":
        answer = get_object_or_404(QuizAnswer, id=request.POST.get("answer_id"))

        answer.human_label = (request.POST.get("human_label") or "").strip()

        human_score_raw = (request.POST.get("human_score") or "").strip()
        if human_score_raw == "":
            answer.human_score = None
        else:
            try:
                answer.human_score = float(human_score_raw)
            except ValueError:
                answer.human_score = None

        answer.save(update_fields=["human_label", "human_score"])
        messages.success(
            request,
            f"{answer.user.username}님의 답안 검수 결과를 저장했습니다.",
        )

        querystring = request.POST.get("querystring", "")
        url = reverse("manage_answers")
        if querystring:
            url = f"{url}?{querystring}"
        return redirect(url)

    answers = (
        QuizAnswer.objects
        .select_related(
            "user",
            "quiz_question",
            "quiz_question__quiz",
            "quiz_question__quiz__lecture",
        )
    )

    search = (request.GET.get("q") or "").strip()
    label = (request.GET.get("label") or "").strip()
    generation = (request.GET.get("generation") or "").strip()
    lecture_id = (request.GET.get("lecture") or "").strip()
    review_state = (request.GET.get("review") or "").strip()

    if search:
        answers = answers.filter(
            Q(user__username__icontains=search)
            | Q(quiz_question__question_text__icontains=search)
            | Q(user_answer__icontains=search)
            | Q(quiz_question__quiz__lecture__title__icontains=search)
        )

    if label:
        answers = answers.filter(predicted_label=label)

    if generation.isdigit():
        answers = answers.filter(
            quiz_question__quiz__generation_number=int(generation)
        )

    if lecture_id.isdigit():
        answers = answers.filter(quiz_question__quiz__lecture_id=int(lecture_id))

    if review_state == "reviewed":
        answers = answers.filter(_REVIEWED_Q)
    elif review_state == "pending":
        answers = answers.exclude(_REVIEWED_Q)

    answers = answers.order_by("-created_at")

    paginator = Paginator(answers, MANAGE_ANSWER_PAGE_SIZE)
    page_obj = paginator.get_page(request.GET.get("page"))

    items = []
    for answer in page_obj:
        raw_score = answer.similarity_score if answer.similarity_score is not None else 0.0
        items.append({
            "obj": answer,
            "percent": round(max(0.0, min(raw_score, 1.0)) * 100, 1),
            "is_reviewed": bool(answer.human_label) or answer.human_score is not None,
        })

    # 페이지 이동 시 필터를 유지하기 위한 쿼리스트링 (page 제외)
    params = request.GET.copy()
    params.pop("page", None)
    base_query = params.urlencode()

    context = {
        "page_obj": page_obj,
        "items": items,
        "search": search,
        "label": label,
        "generation": generation,
        "lecture_id": lecture_id,
        "review_state": review_state,
        "predicted_labels": PREDICTED_LABEL_CHOICES,
        "human_labels": HUMAN_LABEL_CHOICES,
        "lectures": Lecture.objects.order_by("title").values("id", "title"),
        "generations": (
            Quiz.objects.exclude(quiz_text="")
            .values_list("generation_number", flat=True)
            .distinct()
            .order_by("generation_number")
        ),
        "base_query": base_query,
        "current_query": request.GET.urlencode(),
        "reviewed_count": QuizAnswer.objects.filter(_REVIEWED_Q).count(),
        "total_count": QuizAnswer.objects.count(),
    }
    return render(request, "manage/answers.html", context)

@staff_required
def manage_users(request):
    """등록된 사용자 목록 및 UserProfile 수정 페이지."""
    if request.method == "POST":
        user = get_object_or_404(User, id=request.POST.get("user_id"))

        profile, _ = UserProfile.objects.get_or_create(user=user)

        profile.name = (request.POST.get("name") or "").strip()
        profile.phone = (request.POST.get("phone") or "").strip()
        profile.certification = (request.POST.get("certification") or "").strip()
        profile.reason = (request.POST.get("reason") or "").strip()
        profile.interest = (request.POST.get("interest") or "").strip()

        age_raw = (request.POST.get("age") or "").strip()
        if age_raw == "":
            profile.age = None
        else:
            try:
                age_val = int(age_raw)
                profile.age = age_val if age_val > 0 else None
            except ValueError:
                profile.age = None

        profile.save()

        role = (request.POST.get("role") or "").strip()
        ok, err = _apply_user_role(user, role, request.user)
        if not ok:
            messages.error(request, err)
        else:
            messages.success(request, f"{user.username}님의 프로필과 권한을 저장했습니다.")

        querystring = request.POST.get("querystring", "")
        url = reverse("manage_users")
        if querystring:
            url = f"{url}?{querystring}"
        return redirect(url)

    users = (
        User.objects
        .annotate(
            lecture_count=Count("lectures", distinct=True),
            answer_count=Count("quiz_answers", distinct=True),
            display_name=Coalesce("profile__name", Value("")),
        )
        .order_by("-date_joined")
    )

    search = (request.GET.get("q") or "").strip()
    if search:
        users = users.filter(
            Q(username__icontains=search)
            | Q(email__icontains=search)
            | Q(first_name__icontains=search)
            | Q(last_name__icontains=search)
            | Q(profile__name__icontains=search)
        )

    paginator = Paginator(users, MANAGE_USER_PAGE_SIZE)
    page_obj = paginator.get_page(request.GET.get("page"))

    profile_map = {
        p.user_id: p
        for p in UserProfile.objects.filter(
            user_id__in=[u.id for u in page_obj.object_list]
        )
    }

    items = []
    for user in page_obj.object_list:
        profile = profile_map.get(user.id)
        items.append({
            "user": user,
            "profile": profile,
            "display_name": user.display_name,
            "role": _user_role_value(user),
            "is_self": user.pk == request.user.pk,
        })

    params = request.GET.copy()
    params.pop("page", None)
    base_query = params.urlencode()

    context = {
        "page_obj": page_obj,
        "items": items,
        "search": search,
        "base_query": base_query,
        "current_query": request.GET.urlencode(),
        "total_count": User.objects.count(),
        "edit_user_id": (request.GET.get("edit") or "").strip(),
        "user_roles": USER_ROLE_CHOICES,
    }
    return render(request, "manage/users.html", context)
