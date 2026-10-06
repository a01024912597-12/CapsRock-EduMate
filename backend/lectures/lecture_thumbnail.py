"""업로드 영상 썸네일 추출."""

import os

import cv2
from django.conf import settings

from .models import Lecture


def _has_local_video(lecture):
    """업로드 파일 기반 강의이고 디스크에 영상이 있는지 확인."""
    return (
        lecture.source_type == Lecture.SOURCE_FILE
        and bool(lecture.video_file)
        and os.path.exists(lecture.video_file.path)
    )


def capture_lecture_video_thumbnail(lecture, capture_seconds=(2.0, 1.0)):
    """업로드 영상의 1~2초 지점 프레임을 썸네일로 저장하고 URL을 반환한다."""
    if not _has_local_video(lecture):
        return ""

    video_path = lecture.video_file.path
    output_dir = os.path.join(settings.MEDIA_ROOT, "lecture_thumbnails")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{lecture.id}.jpg")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[썸네일 추출 실패] 영상을 열 수 없습니다: {video_path}")
        return ""

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = total_frames / fps if fps > 0 else 0

        for seconds in capture_seconds:
            if duration > 0 and seconds >= duration:
                target_seconds = max(duration - 0.1, 0.0)
            else:
                target_seconds = seconds

            frame_index = max(int(target_seconds * fps), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if not ret or frame is None:
                print(
                    f"[썸네일 추출 재시도] lecture_id={lecture.id}, "
                    f"target={target_seconds:.1f}s"
                )
                continue

            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            if not os.path.exists(output_path):
                continue

            thumbnail_url = f"{settings.MEDIA_URL}lecture_thumbnails/{lecture.id}.jpg"
            print(
                f"[썸네일 추출 완료] lecture_id={lecture.id}, "
                f"sec={target_seconds:.1f}, path={output_path}"
            )
            return thumbnail_url

        print(f"[썸네일 추출 실패] lecture_id={lecture.id}, 사용 가능한 프레임 없음")
        return ""
    except Exception as exc:
        print(f"[썸네일 추출 오류] lecture_id={lecture.id}: {exc}")
        return ""
    finally:
        cap.release()
