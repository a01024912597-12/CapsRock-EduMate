import os

import cv2
import yt_dlp
from django.conf import settings

from ..utils.youtube_utils import extract_youtube_video_id


def download_low_res_video(video_url, output_path):
    """프레임 캡처를 위해 가장 파일 크기가 작고 가벼운 mp4 영상을 다운로드합니다.

    재생목록 URL이 들어오더라도 video_id만 추출해 단일 영상만 다운로드한다.
    """
    video_id = extract_youtube_video_id(video_url)

    if video_id:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "format": "worstvideo[ext=mp4]/worst[ext=mp4]/worst",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        return True

    except Exception as e:
        print(f"[판서 비디오 다운로드 에러] {e}")
        return False


def extract_key_frames_with_timeline(video_path, lecture_id, threshold=0.12, max_frames=10):
    """컬러 히스토그램 비교를 통해 화면 변화가 큰 지점의 프레임을 추출한다.

    보완점:
    - 첫 프레임은 무조건 저장해서 이미지가 0장으로 끝나는 상황을 방지한다.
    - 이후에는 3초마다 검사하면서 변화량이 threshold 이상인 경우만 추가 저장한다.
    """
    output_dir = os.path.join(settings.MEDIA_ROOT, "lecture_frames", str(lecture_id))
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[오픈CV 프레임 추출] 비디오를 열 수 없습니다: {video_path}")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 30

    prev_hist = None
    frame_count = 0
    timeline_frames = []

    print(f"[오픈CV 프레임 추출] 분석 시작 -> 경로: {output_dir}")
    print(f"[오픈CV 프레임 추출 설정] threshold={threshold}, max_frames={max_frames}, fps={fps}")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        ret, frame = cap.read()

        if not ret:
            print(f"[오픈CV 프레임 추출] 더 이상 읽을 프레임이 없습니다. frame_count={frame_count}")
            break

        resized_frame = cv2.resize(frame, (640, 360))
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv_frame],
            [0, 1],
            None,
            [50, 60],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        total_seconds = int(frame_count / fps)
        minutes, seconds = divmod(total_seconds, 60)

        time_str = f"{minutes:02d}:{seconds:02d}"
        safe_time_str = time_str.replace(":", "_")

        should_save = False
        reason = ""

        if prev_hist is None:
            should_save = True
            reason = "첫 프레임"
        else:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            change_amount = 1 - similarity

            if change_amount > threshold:
                should_save = True
                reason = f"화면 변화량={change_amount:.4f}"

        if should_save:
            filename = f"frame_{safe_time_str}.jpg"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(
                filepath,
                resized_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70],
            )

            timeline_frames.append({
                "time_str": time_str,
                "filepath": filepath,
            })

            print(f"[프레임 캡처] {time_str} 지점 화면 저장 완료 / {reason}")

            if len(timeline_frames) >= max_frames:
                print("[오픈CV 프레임 추출] 최대 프레임 개수 도달")
                break

        prev_hist = hist
        frame_count += fps * 3

    cap.release()

    print(f"[오픈CV 프레임 추출] 총 {len(timeline_frames)}개 프레임 추출 완료")

    return timeline_frames


def format_seconds_to_mmss(seconds):
    """초 단위 시간을 MM:SS 또는 H:MM:SS 형식으로 변환한다."""
    try:
        total_seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        total_seconds = 0

    hours, remainder = divmod(total_seconds, 3600)
    minutes, sec = divmod(remainder, 60)

    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"

    return f"{minutes:02d}:{sec:02d}"


def capture_frame_at_timestamp(video_path, lecture_id, seconds, keyword="", output_prefix="core"):
    """요약 핵심 타임라인 시점의 프레임을 직접 캡처한다.

    OpenCV 변화량 기반 추출 프레임이 요약 핵심 시점과 정확히 맞지 않을 수 있으므로,
    요약문에서 추출한 핵심 개념의 타임라인 초 단위 위치를 기준으로 직접 캡처한다.
    """
    if not video_path or not os.path.exists(video_path):
        print(f"[핵심 프레임 캡처] 영상 파일 없음: {video_path}")
        return None

    try:
        target_seconds = max(0, float(seconds))
    except (TypeError, ValueError):
        print(f"[핵심 프레임 캡처] 잘못된 시간값: {seconds}")
        return None

    output_dir = os.path.join(settings.MEDIA_ROOT, "lecture_frames", str(lecture_id))
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[핵심 프레임 캡처] 비디오를 열 수 없습니다: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30

    frame_index = int(target_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()

    # 정확한 프레임 읽기에 실패하면 timestamp 기반으로 한 번 더 시도한다.
    if not ret:
        cap.set(cv2.CAP_PROP_POS_MSEC, target_seconds * 1000)
        ret, frame = cap.read()

    if not ret:
        cap.release()
        print(f"[핵심 프레임 캡처] 프레임 읽기 실패: {target_seconds:.2f}초")
        return None

    resized_frame = cv2.resize(frame, (640, 360))
    time_str = format_seconds_to_mmss(target_seconds)
    safe_time_str = time_str.replace(":", "_")

    filename = f"{output_prefix}_{safe_time_str}.jpg"
    filepath = os.path.join(output_dir, filename)

    ok = cv2.imwrite(
        filepath,
        resized_frame,
        [cv2.IMWRITE_JPEG_QUALITY, 78],
    )

    cap.release()

    if not ok or not os.path.exists(filepath):
        print(f"[핵심 프레임 캡처] 파일 저장 실패: {filepath}")
        return None

    print(f"[핵심 프레임 캡처] {time_str} 지점 저장 완료 / keyword={keyword}")

    return {
        "time_str": time_str,
        "filepath": filepath,
        "seconds": int(round(target_seconds)),
        "keyword": keyword,
        "capture_type": "core_timeline",
    }
