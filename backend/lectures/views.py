from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from urllib.parse import urlparse, parse_qs
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.db.models import Exists, OuterRef, Count, Q, Max
from django.utils import timezone
from .models import Lecture, UserProfile, Quiz, QuizQuestion, QuizAnswer
from google import genai
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util

import yt_dlp
import whisper
import os
import re
import time
import json
import concurrent.futures
import subprocess
import glob
import threading


# =========================
# 기본 경로 및 모델 설정
# =========================

ffmpeg_dir = os.path.join(settings.BASE_DIR, "tools", "ffmpeg", "bin")
if ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + ffmpeg_dir


# Whisper 모델명 관리
# base: 빠름 / small: 속도와 정확도 균형 / medium: 정확도는 높지만 CPU에서 매우 느림
WHISPER_MODEL_NAME = "base"


# 병렬 STT 설정
# 고성능 환경 기준 base + workers=3 조합이 가장 빠르게 측정됨
PARALLEL_MAX_WORKERS = 3

# 오디오 분할 단위: 180초 = 3분
AUDIO_SEGMENT_SECONDS = 180


# Gemini API 설정
GEMINI_MODEL_NAME = getattr(settings, "GEMINI_MODEL_NAME", "gemini-2.5-flash")

_gemini_client = None


def get_gemini_client():
    """Gemini API 클라이언트를 지연 생성한다."""
    global _gemini_client

    if _gemini_client is None:
        api_key = getattr(settings, "GEMINI_API_KEY", "").strip()

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY가 설정되지 않았습니다. "
                "PowerShell에서 $env:GEMINI_API_KEY='키값' 설정 후 서버를 다시 실행하세요."
            )

        _gemini_client = genai.Client(api_key=api_key)

    return _gemini_client


def gemini_generate_text(prompt):
    """Gemini API를 호출하여 텍스트 응답을 반환한다.

    503 UNAVAILABLE, 일시적 과부하 등에 대비해 재시도한다.
    """
    client = get_gemini_client()

    retry_wait_seconds = [3, 7, 15]

    last_error = None

    for attempt, wait_seconds in enumerate(retry_wait_seconds, start=1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )

            text = getattr(response, "text", "")

            if text:
                return text.strip()

            print(f"[Gemini 경고] 응답 text가 비어 있습니다. attempt={attempt}")
            return ""

        except Exception as e:
            last_error = e
            error_text = str(e)

            print(f"[Gemini 호출 에러] attempt={attempt} / {error_text}")

            # 503, UNAVAILABLE, high demand 계열은 잠시 기다렸다 재시도
            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text
                or "temporarily" in error_text
            ):
                print(f"[Gemini 재시도 대기] {wait_seconds}초 후 재시도합니다.")
                time.sleep(wait_seconds)
                continue

            # API 키 오류, 권한 오류 등은 재시도해도 의미 없을 수 있으므로 바로 종료
            raise e

    print(f"[Gemini 최종 실패] 모든 재시도 실패: {last_error}")
    raise last_error


# 각 스레드마다 Whisper 모델을 따로 가지게 하기 위한 저장소
# 전역 모델 하나를 여러 스레드가 동시에 쓰면 Whisper/PyTorch 내부 오류가 날 수 있음
thread_local = threading.local()


def get_whisper_model():
    """현재 스레드 전용 Whisper 모델을 가져온다."""
    if not hasattr(thread_local, "whisper_model"):
        print(f"[Whisper 로딩] thread 전용 모델 로딩 시작: {WHISPER_MODEL_NAME}")
        thread_local.whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
        print(f"[Whisper 로딩] thread 전용 모델 로딩 완료: {WHISPER_MODEL_NAME}")

    return thread_local.whisper_model


# Sentence Transformers 모델명 관리
# 한국어 포함 다국어 문장 유사도 계산용 모델
SENTENCE_TRANSFORMER_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)


# Kiwi 형태소 분석기
kiwi = Kiwi()


# =========================
# 공통 유틸
# =========================

def extract_youtube_video_id(url):
    """유튜브 URL에서 video_id를 추출한다.

    지원 형식:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    """
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


def preprocess_text(text):
    """Whisper 전사 결과를 문장 단위로 정제한다."""
    filler_words = ["음", "어", "그니까", "약간", "이제", "뭐랄까"]

    text = re.sub(r"\s+", " ", text).strip()

    raw_sentences = kiwi.split_into_sents(text)

    processed_sentences = []
    seen_sentences = set()

    for sent in raw_sentences:
        tokens = kiwi.tokenize(sent.text.strip())

        refined_sent = ""

        for token in tokens:
            if token.form not in filler_words:
                if token.tag.startswith("J") or token.tag.startswith("E") or token.tag.startswith("X"):
                    refined_sent += token.form
                else:
                    refined_sent += " " + token.form

        refined_sent = refined_sent.strip()
        refined_sent = re.sub(r"\s+", " ", refined_sent)

        if len(refined_sent) > 10 and refined_sent not in seen_sentences:
            processed_sentences.append(refined_sent)
            seen_sentences.add(refined_sent)

    return processed_sentences


def chunk_text(sentences, chunk_size=10):
    """문장 리스트를 지정 개수 단위의 청크로 묶는다.

    병렬 STT에서는 오디오를 시간 단위로 나누지만,
    Gemini 요약 단계에서는 전사문을 다시 문장 단위로 나누어 요약한다.
    """
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def cleanup_audio_segments(segments=None):
    """분석 후 남은 chunk 파일을 정리한다."""
    base_dir = settings.BASE_DIR

    target_files = []

    if segments:
        target_files.extend(segments)

    target_files.extend(glob.glob(os.path.join(base_dir, "chunk_*.mp3")))

    for file_path in set(target_files):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


def summarize_chunk(chunk_text):
    """단일 청크를 Gemini API로 부분 요약한다.

    팀원 작성 프롬프트를 기반으로 역사 강의의 고유명사와 인과관계를 최대한 보존한다.
    """
    try:
        prompt = f"""너는 역사 교과서를 집필하는 '역사 전공 대학교수' AI이다.
아래는 역사 강의의 일부 구간이다.
이 구간에 등장하는 모든 인물, 연도, 지명, 조약, 국가 이름 등 '고유명사'를 하나도 빠짐없이 추출하고 사건의 인과관계를 요약하라.

[요구사항]
- 인물, 연도, 지명, 조약, 국가 이름 등 구체적인 고유명사를 최대한 빠뜨리지 말 것.
- 인물, 지명 등 고유명사를 표기할 때 괄호 안에 한자나 영어를 병기하지 말고 오직 한글로만 깔끔하게 표기할 것.
- 사건의 원인, 전개 과정, 결과가 드러나도록 정리할 것.
- 모든 문장은 '~했다', '~이다', '~하다' 형식의 객관적인 평어체로 작성할 것.
- 불필요한 잡담이나 반복 표현은 제거할 것.

[출력 형식]
1. 구간 핵심 요약
2. 등장 고유명사 및 핵심 개념
3. 사건의 인과관계
4. 시험 포인트

강의 내용:
{chunk_text}
"""

        result = gemini_generate_text(prompt)

        if not result:
            return f"[부분 요약 생성 실패]\n{chunk_text}"

        return result

    except Exception as e:
        print("Gemini chunk 요약 에러:", e)
        return f"[부분 요약 생성 실패]\n{chunk_text}"


def make_final_summary(chunk_summaries_text):
    """여러 부분 요약을 하나의 최종 요약으로 통합한다.

    Gemini API를 사용하여 역사 강의용 심층 분석 노트를 생성한다.
    """
    try:
        prompt = f"""당신은 역사 교과서를 집필하는 '역사 전공 대학교수' AI입니다.
다음은 유튜브 역사 강의 자막을 구간별로 세밀하게 분석한 내용입니다.

이 내용을 바탕으로, 강의의 중요한 디테일을 놓치지 않고 학생이 복습할 수 있는 심층 분석 노트를 작성해 주세요.
반드시 아래의 [요구사항]과 [출력 양식]을 지켜야 합니다.

[요구사항]
- 강의에 등장하는 모든 인물, 연도, 지명, 조약, 국가 이름 등의 구체적인 고유명사를 최대한 포함할 것.
- 인물, 지명 등 고유명사를 표기할 때 괄호 안에 한자나 영어를 병기하지 말고 오직 한글로만 작성할 것.
- 사건이 일어난 표면적 이유뿐만 아니라, 그 이면에 있는 경제적·정치적 원인까지 인과관계 중심으로 분석할 것.
- 중복 내용은 줄이고, 강의 전체 흐름이 자연스럽게 이어지도록 정리할 것.
- 모든 문장의 끝맺음은 '~했다', '~이다', '~하다' 형식의 객관적인 평어체로 통일할 것.

[출력 양식]

1. 심층 배경 및 전체 요약
- 이 강의가 다루는 시대적 배경과 전체적인 역사적 흐름을 3~5개의 문단으로 서술할 것.

2. 흐름별 상세 전개
- 시대순 또는 사건의 발생 순서대로 [도입] - [전개] - [위기/절정] - [결말/영향] 단계로 나누어 설명할 것.
- 각 단계마다 핵심 사건의 원인과 결과를 명확히 밝힐 것.

3. 꼭 알아야 할 필수 개념 및 고유명사 사전
- 강의에 등장하는 핵심 키워드를 7~10개 정도 선정할 것.
- 각 키워드는 아래 형식으로 정리할 것.

- 키워드 이름
  - 구체적 의미 및 발생 원인
  - 역사적 결과 및 영향

4. 핵심 출제 포인트
- 인과관계, 연도별 변화, 조약의 결과, 국가 간 관계 변화 등 시험에 나올 만한 내용을 불렛포인트로 정리할 것.

강의 전사문 또는 부분 요약 리스트:
{chunk_summaries_text}
"""

        result = gemini_generate_text(prompt)

        if not result:
            return chunk_summaries_text

        return result

    except Exception as e:
        print("Gemini 최종 요약 에러:", e)
        return chunk_summaries_text


# =========================
# 퀴즈 생성 / 파싱 / 저장
# =========================

def generate_quiz(summary_text, generation_number=1, previous_quiz_texts=""):
    """요약문을 기반으로 서술형 예상문제 3개와 모범답안을 Gemini API로 생성한다."""
    try:
        previous_instruction = ""

        if previous_quiz_texts:
            previous_instruction = (
                "\n\n이미 생성된 이전 차수의 문제는 아래와 같다. "
                "새 문제는 이전 문제와 최대한 겹치지 않게 출제하라.\n"
                f"{previous_quiz_texts}"
            )

        prompt = f"""너는 대학 시험 문제 출제자다.
주어진 강의 요약을 바탕으로 복습용 서술형 예상 문제 3개와 각 문제의 모범 답안을 만들어라.

이번 출력은 {generation_number}차 예상 문제이다.

[출제 조건]
- 학생이 시험 대비에 활용할 수 있도록 핵심 개념 중심으로 작성할 것.
- 너무 단순한 암기형 문제보다 원인, 전개 과정, 결과, 의미를 설명하게 하는 문제를 출제할 것.
- 강의 요약에 포함된 인물, 사건, 연도, 조약, 개념을 적절히 반영할 것.
- 이전 차수 문제가 제공된 경우, 이전 문제와 최대한 겹치지 않게 출제할 것.

[출력 형식]
반드시 아래 형식을 그대로 지켜라.
다른 제목, 불릿포인트, 표, 번호 형식을 추가하지 마라.

1. 문제: ...
모범 답안: ...

2. 문제: ...
모범 답안: ...

3. 문제: ...
모범 답안: ...

{previous_instruction}

강의 요약:
{summary_text}
"""

        result = gemini_generate_text(prompt)

        if not result:
            return "퀴즈 생성 중 오류가 발생했습니다."

        return result

    except Exception as e:
        print("Gemini 퀴즈 생성 에러:", e)
        return "퀴즈 생성 중 오류가 발생했습니다."


def parse_quiz_text(quiz_text):
    """AI가 생성한 텍스트를 [문제/모범답안] 구조로 파싱한다."""
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

        QuizQuestion.objects.create(
            quiz=quiz,
            number=number,
            question_text=item.get("question", ""),
            model_answer=item.get("answer", ""),
            explanation=item.get("explanation", ""),
        )

    print(f"[퀴즈 저장] {quiz.generation_number}차 문제 {len(quiz_items)}개를 QuizQuestion에 저장했습니다.")


def ensure_quiz_questions_exist(quiz):
    """기존 quiz_text만 있고 QuizQuestion이 없는 경우 개별 문제를 생성한다."""
    if not quiz.questions.exists() and quiz.quiz_text:
        save_quiz_questions_from_text(quiz, quiz.quiz_text)


# =========================
# 답안 유사도 평가 / 답안 저장
# =========================

def calculate_similarity_score(model_answer, user_answer):
    """모범 답안과 사용자 답안의 의미 유사도를 계산한다.

    반환값:
    - 0.0 ~ 1.0 사이의 유사도 점수
    """
    model_answer = (model_answer or "").strip()
    user_answer = (user_answer or "").strip()

    if not model_answer or not user_answer:
        return 0.0

    try:
        embeddings = sentence_model.encode(
            [model_answer, user_answer],
            convert_to_tensor=True,
        )
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return round(float(score), 4)
    except Exception as e:
        print("[유사도 계산 에러]", e)
        return 0.0


def predict_answer_label(similarity_score):
    """유사도 점수를 기반으로 1차 평가 라벨을 반환한다.

    Sentence Transformers는 최종 채점기가 아니라 의미 유사도 측정 도구이므로,
    단정적인 정답/오답 대신 가능성 중심의 라벨을 사용한다.
    """
    if similarity_score >= 0.75:
        return "정답 가능성 높음"

    if similarity_score >= 0.45:
        return "검토 필요"

    return "오답 가능성 높음"


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
    """POST로 넘어온 사용자 답안을 QuizAnswer 테이블에 저장한다.

    저장 항목:
    - 사용자 답안
    - Sentence Transformers 유사도 점수
    - 시스템 예측 라벨
    """
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

        similarity_score = calculate_similarity_score(
            model_answer=quiz_question.model_answer,
            user_answer=value,
        )
        predicted_label = predict_answer_label(similarity_score)

        QuizAnswer.objects.update_or_create(
            user=user,
            quiz_question=quiz_question,
            defaults={
                "user_answer": value,
                "similarity_score": similarity_score,
                "predicted_label": predicted_label,
            }
        )

        print(
            f"[답안 채점] {quiz.generation_number}차 {quiz_question.number}번 "
            f"유사도={similarity_score}, 판정={predicted_label}"
        )

    # 기존 Quiz.answer JSON 구조도 일단 호환용으로 유지
    quiz.answer = json.dumps(legacy_json_data, ensure_ascii=False)
    quiz.save(update_fields=["answer"])

    print(f"[답안 저장] {quiz.generation_number}차 사용자 답안을 QuizAnswer에 저장했습니다.")
    return legacy_json_data


def build_quiz_items_for_display(quiz_text, quiz, user=None):
    """퀴즈 화면 출력용 데이터 구성.

    우선 QuizQuestion에 저장된 개별 문제를 사용하고,
    사용자 답안은 QuizAnswer 테이블을 우선 사용한다.
    기존 Quiz.answer JSON은 fallback으로만 사용한다.
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

            items.append({
                "number": question.number,
                "question": question.question_text,
                "answer": question.model_answer,
                "explanation": question.explanation,
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

        items.append({
            **item,
            "saved_answer": saved_answer,
        })

    raw_saved = legacy_answer.get("__raw__", "") or ""
    return items, raw_saved


# =========================
# 기본 페이지
# =========================

def home(request):
    """메인(홈) 페이지."""
    return render(request, "home.html")


def login_page(request):
    """로그인 처리 엔드포인트."""
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)

        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"{user.username}님, 환영합니다!")
            return redirect("home")

        messages.error(request, "아이디 또는 비밀번호가 올바르지 않습니다.")
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


@login_required(login_url="home")
def upload_page(request):
    """강의 업로드 페이지."""
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
    """요약 페이지 렌더링."""
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return redirect("upload")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    video_id = (lecture.video_id or "").strip() or extract_youtube_video_id(
        lecture.youtube_url or ""
    ) or ""

    context = {
        "lecture_id": lecture.id,
        "lecture_title": lecture.title,
        "youtube_link": lecture.youtube_url,
        "video_id": video_id,
        "thumbnail_url": lecture.thumbnail_url,
    }

    return render(request, "summary.html", context)


# =========================
# 병렬 STT 핵심 함수
# =========================

def get_audio_segments(video_url):
    """유튜브 오디오를 다운로드한 뒤 일정 시간 단위로 mp3 조각으로 분할한다.

    개선 사항:
    - 유튜브 playlist 방지
    - video_id만 남겨 단일 영상 URL로 정리
    - 이전 chunk 파일 삭제
    - FFmpeg 분할 시 -c copy 대신 16kHz mono로 재인코딩
    - 너무 작은 chunk 파일 제거
    """
    video_id = extract_youtube_video_id(video_url)
    if video_id:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

    base_dir = settings.BASE_DIR
    full_audio_path = os.path.join(base_dir, "temp_full_audio.mp3")
    ffmpeg_bin = os.path.join(base_dir, "tools", "ffmpeg", "bin", "ffmpeg.exe")

    # 이전 작업 파일 삭제
    cleanup_audio_segments()

    if os.path.exists(full_audio_path):
        try:
            os.remove(full_audio_path)
        except OSError:
            pass

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(base_dir, "temp_full_audio"),
        "ffmpeg_location": os.path.dirname(ffmpeg_bin),
        "quiet": False,
    }

    try:
        print("--- [1단계] 전체 오디오 다운로드 시작 ---")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    except Exception as e:
        print(f"--- [에러] 다운로드 실패: {e}")
        return []

    if not os.path.exists(full_audio_path):
        print(f"--- [에러] 원본 오디오 파일이 없습니다: {full_audio_path}")
        return []

    print("--- [2단계] FFmpeg 안정 분할 시작 ---")

    segment_pattern = os.path.join(base_dir, "chunk_%03d.mp3")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", full_audio_path,
        "-f", "segment",
        "-segment_time", str(AUDIO_SEGMENT_SECONDS),
        "-reset_timestamps", "1",

        # Whisper가 읽기 좋은 형태로 재인코딩
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",

        segment_pattern,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"--- [에러] FFmpeg 분할 실패: {e}")

        if e.stderr:
            try:
                print(e.stderr.decode("utf-8", errors="ignore"))
            except Exception:
                pass

        return []

    raw_segments = sorted(glob.glob(os.path.join(base_dir, "chunk_*.mp3")))

    segments = []

    for segment in raw_segments:
        try:
            size = os.path.getsize(segment)
        except OSError:
            size = 0

        # 너무 작은 파일은 빈 chunk일 가능성이 높음
        if size < 10 * 1024:
            print(
                f"--- [경고] 너무 작은 chunk 제거: "
                f"{os.path.basename(segment)} / {size} bytes"
            )

            try:
                os.remove(segment)
            except OSError:
                pass

            continue

        segments.append(segment)

    print(f"--- [3단계] 최종 오디오 조각 개수: {len(segments)}개 ---")

    try:
        if os.path.exists(full_audio_path):
            os.remove(full_audio_path)
    except OSError:
        pass

    return segments


def process_segment_task(segment_path):
    """분할된 오디오 조각 하나를 Whisper STT 처리한다.

    주의:
    - 여기서는 chunk 파일을 삭제하지 않는다.
    - 실패한 chunk를 analyze_video_api()에서 재시도하기 위해 파일을 남긴다.
    """
    segment_name = os.path.basename(segment_path)

    try:
        print(f"[구간 STT 시작] {segment_name}")

        local_model = get_whisper_model()

        stt_start_time = time.time()
        result = local_model.transcribe(
            segment_path,
            language="ko",
            fp16=False,
        )
        stt_duration = time.time() - stt_start_time

        text = result.get("text", "").strip()

        print(
            f"[구간 STT 완료] {segment_name} | "
            f"STT={stt_duration:.2f}초 | 텍스트 길이={len(text)}"
        )

        return {
            "segment": segment_name,
            "segment_path": segment_path,
            "text": text,
            "stt_duration": stt_duration,
            "success": bool(text),
            "error": "",
        }

    except Exception as e:
        print(f"[구간 STT 에러] {segment_path}: {e}")

        return {
            "segment": segment_name,
            "segment_path": segment_path,
            "text": "",
            "stt_duration": 0,
            "success": False,
            "error": str(e),
        }


@login_required(login_url="home")
def analyze_video_api(request):
    """강의 분석 API.

    병렬 STT + Gemini 요약 버전:
    - 유튜브 오디오 다운로드
    - 3분 단위로 오디오 분할
    - 각 구간을 Whisper STT로 병렬 처리
    - 실패한 구간은 순차 재시도
    - 전사 결과를 하나로 합침
    - Gemini API 요약 흐름으로 최종 요약 생성
    - 처리 시간 DB 저장
    """
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return JsonResponse({
            "status": "error",
            "message": "lecture_id가 전달되지 않았습니다.",
        })

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if lecture.summary_text:
        return JsonResponse({
            "status": "success",
            "result": lecture.summary_text,
            "analysis_duration_seconds": lecture.analysis_duration_seconds,
            "stt_duration_seconds": lecture.stt_duration_seconds,
            "summary_duration_seconds": lecture.summary_duration_seconds,
            "whisper_model_name": lecture.whisper_model_name,
        })

    print("\n" + "=" * 60)
    print(f"[병렬 STT + Gemini 요약 시작] 강의 제목: {lecture.title}")
    print(
        f"[설정] Whisper={WHISPER_MODEL_NAME}, "
        f"workers={PARALLEL_MAX_WORKERS}, "
        f"segment={AUDIO_SEGMENT_SECONDS}초, "
        f"Gemini={GEMINI_MODEL_NAME}"
    )
    print("=" * 60)

    analysis_start_time = time.time()
    segments = []

    try:
        print("1. 오디오 다운로드 및 구간 분할 중...")
        segments = get_audio_segments(lecture.youtube_url)

        if not segments:
            analysis_duration = time.time() - analysis_start_time

            lecture.analysis_duration_seconds = analysis_duration
            lecture.stt_duration_seconds = 0
            lecture.summary_duration_seconds = 0
            lecture.whisper_model_name = WHISPER_MODEL_NAME
            lecture.analyzed_at = timezone.now()
            lecture.save(update_fields=[
                "analysis_duration_seconds",
                "stt_duration_seconds",
                "summary_duration_seconds",
                "whisper_model_name",
                "analyzed_at",
            ])

            return JsonResponse({
                "status": "error",
                "message": "오디오 다운로드 또는 분할에 실패했습니다.",
            })

        print(f"2. 병렬 Whisper STT 시작: 총 {len(segments)}개 구간")

        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
            results = list(executor.map(process_segment_task, segments))

        failed_results = [
            result for result in results
            if not result.get("success") or not result.get("text", "").strip()
        ]

        # 실패한 구간만 순차 재시도
        if failed_results:
            print(
                f"--- [재시도] 실패한 STT 구간 {len(failed_results)}개를 "
                f"순차 재시도합니다. ---"
            )

            retry_results = []

            for failed in failed_results:
                segment_path = failed.get("segment_path")

                if not segment_path or not os.path.exists(segment_path):
                    print(f"--- [재시도 불가] 파일 없음: {segment_path}")
                    retry_results.append(failed)
                    continue

                retry_result = process_segment_task(segment_path)

                if retry_result.get("success"):
                    print(f"--- [재시도 성공] {retry_result.get('segment')}")
                else:
                    print(
                        f"--- [재시도 실패] {retry_result.get('segment')} / "
                        f"{retry_result.get('error')}"
                    )

                retry_results.append(retry_result)

            retry_map = {
                item.get("segment"): item
                for item in retry_results
            }

            fixed_results = []

            for result in results:
                segment_name = result.get("segment")

                if segment_name in retry_map:
                    fixed_results.append(retry_map[segment_name])
                else:
                    fixed_results.append(result)

            results = fixed_results

        # 파일명 순서대로 정렬해서 강의 순서 유지
        results = sorted(results, key=lambda x: x.get("segment", ""))

        transcript_parts = []
        total_stt_duration = 0.0
        final_failed_segments = []

        for index, result in enumerate(results, start=1):
            text = result.get("text", "").strip()
            total_stt_duration += result.get("stt_duration", 0)

            if text:
                transcript_parts.append(f"[{index}구간 전사]\n{text}")
            else:
                final_failed_segments.append(result.get("segment", f"{index}구간"))

        # STT 처리 후 chunk 파일 정리
        cleanup_audio_segments(segments)

        if final_failed_segments:
            analysis_duration = time.time() - analysis_start_time

            lecture.analysis_duration_seconds = analysis_duration
            lecture.stt_duration_seconds = total_stt_duration
            lecture.summary_duration_seconds = 0
            lecture.whisper_model_name = f"{WHISPER_MODEL_NAME} / failed_segments"
            lecture.analyzed_at = timezone.now()
            lecture.save(update_fields=[
                "analysis_duration_seconds",
                "stt_duration_seconds",
                "summary_duration_seconds",
                "whisper_model_name",
                "analyzed_at",
            ])

            failed_names = ", ".join(final_failed_segments)

            print(f"--- [최종 실패] STT 실패 구간: {failed_names} ---")

            return JsonResponse({
                "status": "error",
                "message": f"일부 구간 STT에 실패했습니다: {failed_names}",
            })

        full_transcript = "\n\n".join(transcript_parts).strip()

        if not full_transcript:
            analysis_duration = time.time() - analysis_start_time

            lecture.analysis_duration_seconds = analysis_duration
            lecture.stt_duration_seconds = total_stt_duration
            lecture.summary_duration_seconds = 0
            lecture.whisper_model_name = f"{WHISPER_MODEL_NAME} / parallel_stt_failed"
            lecture.analyzed_at = timezone.now()
            lecture.save(update_fields=[
                "analysis_duration_seconds",
                "stt_duration_seconds",
                "summary_duration_seconds",
                "whisper_model_name",
                "analyzed_at",
            ])

            return JsonResponse({
                "status": "error",
                "message": "STT 결과가 비어 있습니다.",
            })

        print("3. 병렬 STT 완료")
        print(f"--- [디버깅] 전체 전사 길이: {len(full_transcript)}")
        print(f"--- [디버깅] 전체 전사 앞부분: {full_transcript[:300]}")

        print("4. Gemini API 요약 시작")
        summary_start_time = time.time()

        # Gemini API 호출 횟수를 줄이기 위해,
        # 짧은 강의는 chunk별 요약을 거치지 않고 전체 전사문을 한 번에 최종 요약한다.
        # 현재 테스트 영상처럼 전체 전사 길이가 몇천 자 수준이면 이 방식이 더 안정적이다.
        if len(full_transcript) <= 12000:
            print("--- [Gemini 요약 방식] 전체 전사문 단일 요약 ---")
            final_summary = make_final_summary(full_transcript)

        else:
            print("--- [Gemini 요약 방식] 긴 전사문 chunk 요약 후 최종 요약 ---")

            sentences = preprocess_text(full_transcript)
            chunks = chunk_text(sentences, chunk_size=10)

            if not chunks:
                chunks = [full_transcript]

            chunk_summaries = []

            for index, chunk in enumerate(chunks, start=1):
                print(f"--- [Gemini chunk 요약] {index}/{len(chunks)} ---")
                summary = summarize_chunk(chunk)
                chunk_summaries.append(summary)

            merged_chunk_summaries = "\n\n".join(chunk_summaries)
            final_summary = make_final_summary(merged_chunk_summaries)

        summary_duration = time.time() - summary_start_time

        result_text = "[AI가 분석한 강의 요약]\n\n" + final_summary

        analysis_duration = time.time() - analysis_start_time

        lecture.summary_text = result_text
        lecture.analysis_duration_seconds = analysis_duration
        lecture.stt_duration_seconds = total_stt_duration
        lecture.summary_duration_seconds = summary_duration
        lecture.whisper_model_name = (
            f"{WHISPER_MODEL_NAME} / "
            f"parallel_stt_workers_{PARALLEL_MAX_WORKERS}_retry / "
            f"gemini_{GEMINI_MODEL_NAME}"
        )
        lecture.analyzed_at = timezone.now()
        lecture.save(update_fields=[
            "summary_text",
            "analysis_duration_seconds",
            "stt_duration_seconds",
            "summary_duration_seconds",
            "whisper_model_name",
            "analyzed_at",
        ])

        print("=" * 60)
        print("[병렬 STT + Gemini 요약 완료]")
        print(f"전체 분석 시간: {analysis_duration:.2f}초")
        print(f"구간별 STT 시간 합계: {total_stt_duration:.2f}초")
        print(f"Gemini 요약 시간: {summary_duration:.2f}초")
        print(f"workers: {PARALLEL_MAX_WORKERS}")
        print("=" * 60 + "\n")

        return JsonResponse({
            "status": "success",
            "result": result_text,
            "analysis_duration_seconds": analysis_duration,
            "stt_duration_seconds": total_stt_duration,
            "summary_duration_seconds": summary_duration,
            "whisper_model_name": lecture.whisper_model_name,
        })

    except Exception as e:
        cleanup_audio_segments(segments)

        analysis_duration = time.time() - analysis_start_time

        lecture.analysis_duration_seconds = analysis_duration
        lecture.stt_duration_seconds = 0
        lecture.summary_duration_seconds = 0
        lecture.whisper_model_name = f"{WHISPER_MODEL_NAME} / parallel_error / gemini_{GEMINI_MODEL_NAME}"
        lecture.analyzed_at = timezone.now()
        lecture.save(update_fields=[
            "analysis_duration_seconds",
            "stt_duration_seconds",
            "summary_duration_seconds",
            "whisper_model_name",
            "analyzed_at",
        ])

        print(f"[병렬 STT + Gemini 요약 시스템 에러] {str(e)}")

        return JsonResponse({
            "status": "error",
            "message": str(e),
        })


# =========================
# 히스토리 / 퀴즈 / 답안 / 피드백
# =========================

@login_required(login_url="home")
def history_page(request):
    """내 강의 히스토리 페이지."""
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
    """예상문제 페이지."""
    lecture_id = request.GET.get("lecture_id")

    if request.method == "POST":
        lecture_id = lecture_id or request.POST.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if not lecture.summary_text:
        return redirect(f"/summary/?lecture_id={lecture.id}")

    quizzes = Quiz.objects.filter(lecture=lecture).exclude(quiz_text="").order_by("generation_number", "id")

    if request.method == "POST" and request.POST.get("save_user_answers"):
        gen_raw = request.POST.get("generation")

        if not gen_raw or not str(gen_raw).isdigit():
            return redirect("history")

        gen = int(gen_raw)
        quiz_to_update = get_object_or_404(Quiz, lecture=lecture, generation_number=gen)

        save_user_answers_to_quiz_answer(
            user=request.user,
            quiz=quiz_to_update,
            post_data=request.POST,
        )

        return redirect(f"/quiz/?lecture_id={lecture.id}&generation={gen}")

    if request.method == "POST":
        max_generation = quizzes.aggregate(max_number=Max("generation_number"))["max_number"] or 0
        next_generation = max_generation + 1

        previous_quiz_texts = "\n\n".join([quiz.quiz_text for quiz in quizzes])

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
            explanation="",
            quiz_text=quiz_text,
        )

        save_quiz_questions_from_text(new_quiz, quiz_text)

        return redirect(f"/quiz/?lecture_id={lecture.id}&generation={next_generation}")

    if not quizzes.exists():
        quiz_text = generate_quiz(
            lecture.summary_text,
            generation_number=1,
            previous_quiz_texts="",
        )

        new_quiz = Quiz.objects.create(
            lecture=lecture,
            generation_number=1,
            question=quiz_text,
            answer="",
            explanation="",
            quiz_text=quiz_text,
        )

        save_quiz_questions_from_text(new_quiz, quiz_text)

        return redirect(f"/quiz/?lecture_id={lecture.id}&generation=1")

    selected_generation = request.GET.get("generation")

    if selected_generation:
        selected_quiz = quizzes.filter(generation_number=selected_generation).first()
    else:
        selected_quiz = quizzes.order_by("-generation_number", "-id").first()

    if not selected_quiz:
        selected_quiz = quizzes.first()

    quiz_text = selected_quiz.quiz_text
    quiz_items, raw_user_answer = build_quiz_items_for_display(
        quiz_text=quiz_text,
        quiz=selected_quiz,
        user=request.user,
    )

    return render(request, "quiz.html", {
        "lecture": lecture,
        "quizzes": quizzes,
        "selected_quiz": selected_quiz,
        "quiz_text": quiz_text,
        "quiz_items": quiz_items,
        "raw_user_answer": raw_user_answer,
    })


@login_required(login_url="home")
def quiz_answer_page(request):
    """모범 답안, 입력한 답안 확인 가능."""
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    if not lecture.summary_text:
        return redirect(f"/summary/?lecture_id={lecture.id}")

    quizzes = Quiz.objects.filter(lecture=lecture).exclude(quiz_text="").order_by("generation_number", "id")

    if not quizzes.exists():
        return redirect(f"/quiz/?lecture_id={lecture.id}")

    selected_generation = request.GET.get("generation")

    if selected_generation:
        selected_quiz = quizzes.filter(generation_number=selected_generation).first()
    else:
        selected_quiz = quizzes.order_by("-generation_number", "-id").first()

    if not selected_quiz:
        selected_quiz = quizzes.first()

    quiz_text = selected_quiz.quiz_text
    quiz_items, raw_user_answer = build_quiz_items_for_display(
        quiz_text=quiz_text,
        quiz=selected_quiz,
        user=request.user,
    )

    return render(request, "quiz_answer.html", {
        "lecture": lecture,
        "quizzes": quizzes,
        "selected_quiz": selected_quiz,
        "quiz_text": quiz_text,
        "quiz_items": quiz_items,
        "raw_user_answer": raw_user_answer,
    })


@login_required(login_url="home")
def feedback_page(request):
    """학습 피드백 페이지.

    강의별 사용자 답안, 유사도 점수, 시스템 1차 판단,
    사람 검토 결과를 모아 학습 분석 결과로 보여준다.
    """
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return redirect("history")

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    has_summary = bool(lecture.summary_text)
    has_quiz = Quiz.objects.filter(lecture=lecture).exclude(quiz_text="").exists()

    answers = (
        QuizAnswer.objects
        .filter(
            user=request.user,
            quiz_question__quiz__lecture=lecture,
        )
        .select_related(
            "quiz_question",
            "quiz_question__quiz",
        )
        .order_by(
            "quiz_question__quiz__generation_number",
            "quiz_question__number",
            "created_at",
        )
    )

    total_answer_count = answers.count()

    correct_like_count = 0
    review_needed_count = 0
    wrong_like_count = 0

    reviewed_count = 0
    review_agree_count = 0

    similarity_sum = 0.0
    answer_items = []

    def normalize_label(label):
        """시스템 라벨과 사람 라벨을 비교하기 위한 공통 분류."""
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
        """human_label이 없을 때 human_score를 기준으로 라벨을 추정한다."""
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

    for answer in answers:
        score = answer.similarity_score if answer.similarity_score is not None else 0.0
        similarity_sum += score

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

        answer_items.append({
            "id": answer.id,
            "generation_number": answer.quiz_question.quiz.generation_number,
            "question_number": answer.quiz_question.number,
            "question_text": answer.quiz_question.question_text,
            "model_answer": answer.quiz_question.model_answer,
            "user_answer": answer.user_answer,
            "similarity_score": score,
            "similarity_percent": round(score * 100, 1),
            "predicted_label": predicted_label,
            "human_score": human_score,
            "human_label": human_label,
            "is_reviewed": is_reviewed,
            "created_at": answer.created_at,
        })

    if total_answer_count > 0:
        average_similarity = round((similarity_sum / total_answer_count) * 100, 1)
    else:
        average_similarity = 0

    if reviewed_count > 0:
        review_agreement_rate = round((review_agree_count / reviewed_count) * 100, 1)
    else:
        review_agreement_rate = None

    return render(request, "feedback.html", {
        "lecture": lecture,
        "has_summary": has_summary,
        "has_quiz": has_quiz,

        "answer_items": answer_items,
        "total_answer_count": total_answer_count,
        "average_similarity": average_similarity,

        "correct_like_count": correct_like_count,
        "review_needed_count": review_needed_count,
        "wrong_like_count": wrong_like_count,

        "reviewed_count": reviewed_count,
        "review_agree_count": review_agree_count,
        "review_agreement_rate": review_agreement_rate,
    })


def test_api(request):
    """프론트-백엔드 연결 테스트용 API."""
    return JsonResponse(
        {"message": "백엔드 연결 성공"},
        json_dumps_params={"ensure_ascii": False},
    )