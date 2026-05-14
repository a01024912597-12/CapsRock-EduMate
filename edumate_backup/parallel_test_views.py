from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from urllib.parse import urlparse, parse_qs
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.db.models import Exists, OuterRef, Count, Q, Max
from .models import Lecture, UserProfile, Quiz

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from kiwipiepy import Kiwi
import yt_dlp
import whisper
import os
import re
import time
import concurrent.futures
import subprocess
import glob

ffmpeg_dir = os.path.join(settings.BASE_DIR, "tools", "ffmpeg", "bin")
if ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

# 기존 base보다 정확도 향상을 기대할 수 있는 small 모델 사용
# 더 높은 정확도를 원하면 "medium"으로 변경 가능하지만, CPU 환경에서는 처리 시간이 크게 늘어날 수 있음
model = whisper.load_model("small")

# Kiwi 형태소 분석기
kiwi = Kiwi()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KoBART 모델과 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
summary_model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization').to(device)


def extract_youtube_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


def summarize_with_kobart(text):
    # 입력 텍스트가 유효한지 먼저 체크
    if not text or len(text.strip()) < 5:
        print("[경고] 요약할 텍스트가 너무 짧거나 비어있습니다.")
        return "요약할 수 있는 충분한 내용이 없습니다."

    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    # 토큰화된 결과가 0개인지 체크 (에러 방지 핵심)
    if input_ids.numel() == 0:
        return "텍스트 처리 중 오류가 발생했습니다."

    max_pos = 1024
    summary_list = []

    for i in range(0, input_ids.shape[1], max_pos):
        chunk = input_ids[:, i: i + max_pos]

        if chunk.numel() == 0:
            continue

        output_ids = summary_model.generate(
            chunk,
            num_beams=4,
            max_length=250,
            eos_token_id=tokenizer.eos_token_id
        )
        summary_list.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return " ".join(summary_list)


def preprocess_text(text):
    filler_words = ["음", "어", "그니까", "약간", "이제", "뭐랄까"]

    text = re.sub(r"\s+", " ", text).strip()

    # Kiwi의 문장 분리 기능 사용
    raw_sentences = kiwi.split_into_sents(text)

    processed_sentences = []
    seen_sentences = set()

    for sent in raw_sentences:
        tokens = kiwi.tokenize(sent.text.strip())

        refined_sent = ""

        for token in tokens:
            if token.form not in filler_words:
                # 조사, 어미, 접사는 앞 단어에 붙이고 나머지는 띄어쓰기 유지
                if token.tag.startswith("J") or token.tag.startswith("E") or token.tag.startswith("X"):
                    refined_sent += token.form
                else:
                    refined_sent += " " + token.form

        refined_sent = refined_sent.strip()
        refined_sent = re.sub(r"\s+", " ", refined_sent)

        # 중복 문장 제거 및 너무 짧은 문장 제외
        if len(refined_sent) > 10 and refined_sent not in seen_sentences:
            key_words = []

            for key in key_words:
                if key in refined_sent:
                    refined_sent = refined_sent.replace(key, f"**{key}**")

            processed_sentences.append(refined_sent)
            seen_sentences.add(refined_sent)

    return processed_sentences


def parse_quiz_text(quiz_text):
    quiz_items = []

    if not quiz_text:
        return quiz_items

    pattern = r"(\d+)\.\s*문제\s*:\s*(.*?)(?:\n|\r\n)\s*모범\s*답안\s*:\s*(.*?)(?=\n\s*\d+\.\s*문제\s*:|\Z)"
    matches = re.findall(pattern, quiz_text, re.DOTALL)

    for number, question, answer in matches:
        quiz_items.append({
            "number": number.strip(),
            "question": question.strip(),
            "answer": answer.strip(),
        })

    return quiz_items


def home(request):
    return render(request, "home.html")


def login_page(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"{user.username}님, 환영합니다!")
            return redirect("home")
        else:
            messages.error(request, "아이디 또는 비밀번호가 올바르지 않습니다.")
            return redirect("home")

    return redirect("home")


def logout_page(request):
    logout(request)
    return redirect("home")


def signup_page(request):
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
        else:
            messages.error(request, "회원가입에 실패했습니다. 입력값을 다시 확인해주세요.")
            return render(request, "signup.html", {"form": form})

    form = UserCreationForm()
    return render(request, "signup.html", {"form": form})


@login_required(login_url="home")
def upload_page(request):
    if request.method == "POST":
        title = request.POST.get("title", "")
        youtube_link = request.POST.get("youtube_link", "")

        video_id = extract_youtube_video_id(youtube_link)
        thumbnail_url = ""

        if video_id:
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"

        lecture = Lecture.objects.create(
            user=request.user,
            title=title,
            youtube_url=youtube_link,
            video_id=video_id or "",
            thumbnail_url=thumbnail_url,
        )

        return redirect(f"/summary/?lecture_id={lecture.id}")

    return render(request, "upload.html")


@login_required(login_url="home")
def summary_page(request):
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return redirect("upload")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    context = {
        "lecture_id": lecture.id,
        "lecture_title": lecture.title,
        "youtube_link": lecture.youtube_url,
        "thumbnail_url": lecture.thumbnail_url,
    }
    return render(request, "summary.html", context)


# 영상을 받고 3분 단위로 쪼개기
def get_audio_segments(video_url):
    try:
        current_file_path = os.path.abspath(__file__)
        computed_base_dir = os.path.dirname(os.path.dirname(current_file_path))

        BASE_DIR = getattr(settings, 'BASE_DIR', computed_base_dir)
    except Exception:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    full_audio_path = os.path.join(BASE_DIR, "temp_full_audio.mp3")
    ffmpeg_bin = os.path.join(BASE_DIR, "tools", "ffmpeg", "bin", "ffmpeg.exe")

    # 이전 작업 파일 삭제
    for f in glob.glob(os.path.join(BASE_DIR, "chunk_*.mp3")):
        os.remove(f)
    if os.path.exists(full_audio_path):
        os.remove(full_audio_path)

    # 전체 오디오 다운로드 (기존 yt_dlp 로직 활용)
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": os.path.join(BASE_DIR, "temp_full_audio"),
        "ffmpeg_location": os.path.dirname(ffmpeg_bin),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"--- [에러] 다운로드 실패: {e}")
        return []

        # 3. 오디오 분할 (3분 단위)
    if not os.path.exists(full_audio_path):
        print(f"--- [에러] 원본 파일이 없습니다: {full_audio_path}")
        return []

    print("--- [2단계] FFmpeg 분할 시작...")
    segment_pattern = os.path.join(BASE_DIR, "chunk_%03d.mp3")

    # subprocess를 사용하여 공백이 포함된 경로도 안전하게 처리
    cmd = [
        ffmpeg_bin, "-i", full_audio_path,
        "-f", "segment", "-segment_time", "180",
        "-c", "copy", segment_pattern
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"--- [에러] FFmpeg 실행 실패: {e}")
        return []

    # 4. 결과 리스트 반환
    segments = sorted(glob.glob(os.path.join(BASE_DIR, "chunk_*.mp3")))
    print(f"--- [3단계] 최종 조각 개수: {len(segments)}개")

    if os.path.exists(full_audio_path):
        os.remove(full_audio_path)

    return segments


# 쪼개진 파일 하나를 처리
def process_segment_task(segment_path):
    try:
        # Whisper STT 실행[cite: 3]
        result = model.transcribe(segment_path, language="ko")
        text = result.get("text", "")

        # 요약 모델 실행
        sentences = preprocess_text(text)
        clean_text = " ".join(sentences)
        summary = summarize_with_kobart(clean_text)

        # 사용한 조각 파일 삭제
        if os.path.exists(segment_path): os.remove(segment_path)
        return summary
    except Exception as e:
        return f"구간 처리 에러: {str(e)}"


@login_required(login_url="home")
def analyze_video_api(request):
    lecture_id = request.GET.get("lecture_id")
    if not lecture_id:
        return JsonResponse({"status": "error", "message": "lecture_id가 없습니다."})

    # DB에서 강의 정보 가져오기
    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    # 이미 요약 결과가 저장되어 있다면 바로 반환
    if lecture.summary_text:
        return JsonResponse({"status": "success", "result": lecture.summary_text})

    total_start_time = time.time()
    print(f"\n" + "=" * 50)
    print(f"[병렬 분석 시작] 강의 제목: {lecture.title}")

    try:
        # 전체 오디오 다운로드 및 3분 단위 분할
        print("1. 오디오 다운로드 및 조각 분리 중...")
        segments = get_audio_segments(lecture.youtube_url)

        # ThreadPoolExecutor를 사용한 병렬 처리
        print(f"2. 병렬 STT 및 요약 시작 (총 {len(segments)}개 조각)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            summaries = list(executor.map(process_segment_task, segments))

        # 분할된 요약 결과들을 하나로 합침
        final_summary = "\n\n".join(summaries)

        # 결과값 DB 저장
        lecture.summary_text = final_summary
        lecture.save()

        total_duration = time.time() - total_start_time
        print(f"   => 전체 분석 완료 (소요 시간: {total_duration:.2f}초)")
        print("=" * 50 + "\n")

        return JsonResponse({"status": "success", "result": final_summary})

    except Exception as e:
        print(f"[시스템 에러] {str(e)}")
        return JsonResponse({"status": "error", "message": str(e)})


@login_required(login_url="home")
def history_page(request):
    quiz_exists = Quiz.objects.filter(lecture=OuterRef("pk")).exclude(quiz_text="")

    lectures = (
        Lecture.objects
        .filter(user=request.user)
        .annotate(has_quiz=Exists(quiz_exists))
        .annotate(quiz_count=Count("quizzes", filter=Q(quizzes__quiz_text__gt=""), distinct=True))
        .order_by("-created_at")
    )

    return render(request, "history.html", {"lectures": lectures})


@login_required(login_url="home")
def quiz_page(request):
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if not lecture.summary_text:
        return redirect(f"/summary/?lecture_id={lecture.id}")

    quizzes = Quiz.objects.filter(lecture=lecture).exclude(quiz_text="").order_by("generation_number", "id")

    if request.method == "POST":
        max_generation = quizzes.aggregate(max_number=Max("generation_number"))["max_number"] or 0
        next_generation = max_generation + 1

        previous_quiz_texts = "\n\n".join([quiz.quiz_text for quiz in quizzes])

        quiz_text = generate_quiz(
            lecture.summary_text,
            generation_number=next_generation,
            previous_quiz_texts=previous_quiz_texts,
        )

        Quiz.objects.create(
            lecture=lecture,
            generation_number=next_generation,
            question=quiz_text,
            answer="",
            explanation="",
            quiz_text=quiz_text,
        )

        return redirect(f"/quiz/?lecture_id={lecture.id}&generation={next_generation}")

    if not quizzes.exists():
        quiz_text = generate_quiz(
            lecture.summary_text,
            generation_number=1,
            previous_quiz_texts="",
        )

        Quiz.objects.create(
            lecture=lecture,
            generation_number=1,
            question=quiz_text,
            answer="",
            explanation="",
            quiz_text=quiz_text,
        )

        return redirect(f"/quiz/?lecture_id={lecture.id}&generation=1")

    selected_generation = request.GET.get("generation")

    if selected_generation:
        selected_quiz = quizzes.filter(generation_number=selected_generation).first()
    else:
        selected_quiz = quizzes.order_by("-generation_number", "-id").first()

    if not selected_quiz:
        selected_quiz = quizzes.first()

    quiz_text = selected_quiz.quiz_text
    quiz_items = parse_quiz_text(quiz_text)

    return render(request, "quiz.html", {
        "lecture": lecture,
        "quizzes": quizzes,
        "selected_quiz": selected_quiz,
        "quiz_text": quiz_text,
        "quiz_items": quiz_items,
    })


@login_required(login_url="home")
def feedback_page(request):
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    has_summary = bool(lecture.summary_text)
    has_quiz = Quiz.objects.filter(lecture=lecture).exclude(quiz_text="").exists()

    return render(request, "feedback.html", {
        "lecture": lecture,
        "has_summary": has_summary,
        "has_quiz": has_quiz,
    })


def test_api(request):
    return JsonResponse(
        {"message": "백엔드 연결 성공"},
        json_dumps_params={"ensure_ascii": False},
    )
