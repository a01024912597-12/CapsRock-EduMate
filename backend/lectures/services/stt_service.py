import os
import time
import glob
import threading
import subprocess

import yt_dlp
import whisper
from django.conf import settings

from ..utils.youtube_utils import extract_youtube_video_id

# FFmpeg 경로 설정
ffmpeg_dir = os.path.join(settings.BASE_DIR, "tools", "ffmpeg", "bin")
if ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

# Whisper 모델명 관리
WHISPER_MODEL_NAME = "base"

# 오디오 분할 단위: 180초 = 3분
AUDIO_SEGMENT_SECONDS = 180

# 각 스레드마다 Whisper 모델을 따로 가지게 하기 위한 저장소
thread_local = threading.local()


def get_whisper_model():
    """현재 스레드 전용 Whisper 모델을 가져온다.

    업로드 페이지에서 선택한 모델이 바뀐 경우,
    기존 thread_local 모델을 재사용하지 않고 새 모델을 로딩한다.
    """
    current_model_name = WHISPER_MODEL_NAME

    cached_model_name = getattr(thread_local, "whisper_model_name", None)

    if (
        not hasattr(thread_local, "whisper_model")
        or cached_model_name != current_model_name
    ):
        print(f"[Whisper 로딩] thread 전용 모델 로딩 시작: {current_model_name}")
        thread_local.whisper_model = whisper.load_model(current_model_name)
        thread_local.whisper_model_name = current_model_name
        print(f"[Whisper 로딩] thread 전용 모델 로딩 완료: {current_model_name}")

    return thread_local.whisper_model

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

def _format_timestamp_mmss(seconds):
    """초 단위 시간을 m:ss 형식으로 변환한다."""
    s = max(0, int(round(float(seconds))))
    m, sec = divmod(s, 60)
    return f"{m:d}:{sec:02d}"

def _whisper_segments_to_plain(transcribe_result):
    """Whisper transcribe 결과의 segments를 JSON 직렬화 가능한 dict 리스트로 정규화한다."""
    out = []

    for seg in transcribe_result.get("segments") or []:
        if isinstance(seg, dict):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = (seg.get("text") or "").strip()
        else:
            start = float(getattr(seg, "start", 0.0))
            end = float(getattr(seg, "end", start))
            text = (getattr(seg, "text", "") or "").strip()

        if not text:
            continue

        out.append({
            "start": float(start),
            "end": float(end),
            "text": text,
        })

    return out

def _segment_span_chunk_string(segments):
    """연속 세그먼트 묶음을 요약 입력용 문자열로 만든다."""
    if not segments:
        return ""

    t0 = segments[0]["start"]
    t1 = segments[-1]["end"]

    header = f"[{_format_timestamp_mmss(t0)} ~ {_format_timestamp_mmss(t1)}]"
    body = "\n".join(s["text"] for s in segments)

    return f"{header}\n{body}"

def chunks_from_whisper_segments(
    segments,
    segments_per_chunk=12,
    max_chars_per_chunk=4500,
):
    """Whisper 세그먼트를 시간대가 유지된 청크 문자열 목록으로 만든다."""
    if not segments:
        return []

    chunks = []
    buf = []
    buf_chars = 0

    for seg in segments:
        piece = seg["text"]

        if buf and (
            len(buf) >= segments_per_chunk
            or buf_chars + len(piece) + 1 > max_chars_per_chunk
        ):
            chunks.append(_segment_span_chunk_string(buf))
            buf = []
            buf_chars = 0

        buf.append(seg)
        buf_chars += len(piece) + 1

    if buf:
        chunks.append(_segment_span_chunk_string(buf))

    return chunks

def _ffmpeg_executable():
    """프로젝트 내 FFmpeg가 있으면 우선 사용하고, 없으면 PATH의 ffmpeg를 사용한다."""
    local_ffmpeg = os.path.join(settings.BASE_DIR, "tools", "ffmpeg", "bin", "ffmpeg.exe")

    if os.path.exists(local_ffmpeg):
        return local_ffmpeg

    return "ffmpeg"


def _split_full_audio_into_segments(full_audio_path):
    """mp3 원본을 일정 시간 단위 chunk로 분할한다."""
    base_dir = settings.BASE_DIR
    ffmpeg_bin = _ffmpeg_executable()

    cleanup_audio_segments()

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


def _download_youtube_audio_to_mp3(video_url, full_audio_path):
    """유튜브 URL에서 오디오를 mp3로 다운로드한다."""
    video_id = extract_youtube_video_id(video_url)

    if video_id:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

    ffmpeg_bin = _ffmpeg_executable()

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
        "outtmpl": os.path.splitext(full_audio_path)[0],
        "ffmpeg_location": os.path.dirname(ffmpeg_bin) if os.path.isabs(ffmpeg_bin) else None,
        "quiet": False,
    }

    if not ydl_opts["ffmpeg_location"]:
        ydl_opts.pop("ffmpeg_location")

    try:
        print("--- [1단계] 전체 오디오 다운로드 시작 ---")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    except Exception as e:
        print(f"--- [에러] 다운로드 실패: {e}")
        return False

    return os.path.exists(full_audio_path)


def _extract_local_video_audio_to_mp3(video_path, full_audio_path):
    """로컬 영상 파일에서 FFmpeg로 오디오를 mp3로 추출한다."""
    ffmpeg_bin = _ffmpeg_executable()

    if os.path.exists(full_audio_path):
        try:
            os.remove(full_audio_path)
        except OSError:
            pass

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        full_audio_path,
    ]

    try:
        print("--- [1단계] 로컬 영상 오디오 추출 시작 ---")
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"--- [에러] 로컬 오디오 추출 실패: {e}")

        if e.stderr:
            try:
                print(e.stderr.decode("utf-8", errors="ignore"))
            except Exception:
                pass

        return False

    return os.path.exists(full_audio_path)


def _build_audio_segments(source, audio_extract_func):
    """오디오 준비 함수가 성공하면 공통 분할 로직을 실행한다."""
    full_audio_path = os.path.join(settings.BASE_DIR, "temp_full_audio.mp3")

    if not audio_extract_func(source, full_audio_path):
        return []

    return _split_full_audio_into_segments(full_audio_path)


def get_audio_segments(video_url):
    """유튜브 오디오를 다운로드한 뒤 일정 시간 단위로 mp3 조각으로 분할한다."""
    return _build_audio_segments(video_url, _download_youtube_audio_to_mp3)


def get_audio_segments_from_local_video(video_path):
    """로컬 영상 파일에서 오디오를 추출한 뒤 mp3 조각으로 분할한다."""
    return _build_audio_segments(video_path, _extract_local_video_audio_to_mp3)


def lecture_has_local_video(lecture):
    """업로드 파일 기반 강의이고 디스크에 영상이 있는지 확인한다."""
    return (
        getattr(lecture, "source_type", "youtube") == "file"
        and bool(getattr(lecture, "video_file", None))
        and os.path.exists(lecture.video_file.path)
    )


def get_audio_segments_for_lecture(lecture):
    """강의 소스 유형에 맞게 오디오 구간 목록을 반환한다."""
    if lecture_has_local_video(lecture):
        return get_audio_segments_from_local_video(lecture.video_file.path)

    if getattr(lecture, "youtube_url", ""):
        return get_audio_segments(lecture.youtube_url)

    return []

def process_segment_task(args):
    """분할된 오디오 조각 하나를 Whisper STT 처리하고 offset을 적용한 세그먼트를 반환한다."""
    segment_path, offset_seconds = args
    segment_name = os.path.basename(segment_path)

    try:
        print(f"[구간 STT 시작] {segment_name} / offset={offset_seconds}s")

        local_model = get_whisper_model()

        stt_start_time = time.time()
        result = local_model.transcribe(
            segment_path,
            language="ko",
            fp16=False,
        )
        stt_duration = time.time() - stt_start_time

        text = result.get("text", "").strip()
        segments = _whisper_segments_to_plain(result)

        for seg in segments:
            seg["start"] += offset_seconds
            seg["end"] += offset_seconds

        print(
            f"[구간 STT 완료] {segment_name} | "
            f"STT={stt_duration:.2f}초 | 텍스트 길이={len(text)} | 세그먼트={len(segments)}개"
        )

        return {
            "segment": segment_name,
            "segment_path": segment_path,
            "text": text,
            "segments": segments,
            "stt_duration": stt_duration,
            "success": bool(text or segments),
            "error": "",
        }

    except Exception as e:
        print(f"[구간 STT 에러] {segment_path}: {e}")

        return {
            "segment": segment_name,
            "segment_path": segment_path,
            "text": "",
            "segments": [],
            "stt_duration": 0,
            "success": False,
            "error": str(e),
        }
