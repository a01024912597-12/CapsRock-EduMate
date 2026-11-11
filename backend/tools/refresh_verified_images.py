"""기존 승인 이미지만 고화질 영상에서 다시 캡처한다. 요약/STT/API는 호출하지 않는다.

Usage:
    python tools/refresh_verified_images.py 144
    python tools/refresh_verified_images.py 144 --dry-run
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from django.conf import settings
from lectures.models import Lecture
from lectures.services.media_service import download_low_res_video


def _time_from_filename(name):
    """verified_0_10_52.jpg -> 652초, verified_1_01_02_03.jpg -> 3723초."""
    match = re.fullmatch(r"verified_\d+_(\d+(?:_\d+){1,2})\.jpg", name, re.I)
    if not match:
        return None
    parts = [int(x) for x in match.group(1).split("_")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def refresh(lecture_id, dry_run=False):
    lecture = Lecture.objects.get(pk=lecture_id)
    output_dir = Path(settings.MEDIA_ROOT) / "lecture_frames" / str(lecture_id)
    if not output_dir.is_dir():
        print(f"이미지 폴더가 없습니다: {output_dir}")
        return 0

    linked = lecture.summary_text or ""
    items = [(file, _time_from_filename(file.name)) for file in output_dir.glob("verified_*.jpg")]
    items = [(file, sec) for file, sec in items if sec is not None and file.name in linked]
    if not items:
        print("요약문에 연결된 기존 승인 이미지가 없습니다. 변경하지 않습니다.")
        return 0

    print(f"강의 ID {lecture_id}: 고화질로 교체할 이미지 {len(items)}장")
    for file, sec in items:
        print(f"  {file.name} ({sec}초)")
    if dry_run:
        print("--dry-run: 실제 파일을 변경하지 않았습니다.")
        return 0

    with tempfile.TemporaryDirectory(prefix=f"edumate_hd_{lecture_id}_") as temp_dir:
        video_path = ""
        if lecture.video_file:
            try:
                candidate = lecture.video_file.path
                if os.path.isfile(candidate):
                    video_path = candidate
                    print("업로드된 원본 영상을 사용합니다.")
            except (OSError, ValueError):
                pass
        if not video_path:
            if not lecture.youtube_url:
                print("유튜브 URL 또는 원본 영상이 없습니다. 교체하지 않습니다.")
                return 0
            video_path = os.path.join(temp_dir, "frames_source.mp4")
            if not download_low_res_video(lecture.youtube_url, video_path):
                print("영상 다운로드 실패. 기존 이미지는 그대로 유지합니다.")
                return 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("영상을 열 수 없습니다. 기존 이미지는 그대로 유지합니다.")
            return 0
        prepared = []
        try:
            for image_path, sec in items:
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                ok, frame = cap.read()
                if not ok:
                    print(f"{image_path.name}: 캡처 실패, 건너뜀")
                    continue
                h, w = frame.shape[:2]
                factor = min(1.0, 1920 / w, 1080 / h)
                if factor < 1:
                    frame = cv2.resize(
                        frame, (max(1, round(w * factor)), max(1, round(h * factor))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    prepared.append((image_path, encoded.tobytes(), frame.shape[1], frame.shape[0]))
        finally:
            cap.release()

        if not prepared:
            print("새 캡처가 없어 기존 이미지는 그대로 유지합니다.")
            return 0
        backup = output_dir / ("_backup_before_hd_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        backup.mkdir(parents=True, exist_ok=True)
        for image_path, data, width, height in prepared:
            shutil.copy2(image_path, backup / image_path.name)
            temporary_file = image_path.with_name(image_path.name + ".tmp")
            try:
                temporary_file.write_bytes(data)
                os.replace(temporary_file, image_path)
                print(f"교체 완료: {image_path.name} -> {width}x{height}")
            finally:
                if temporary_file.exists():
                    temporary_file.unlink()
        print(f"기존 이미지 백업: {backup}")
        print("요약문과 DB는 변경하지 않았습니다.")
        return len(prepared)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="강의의 검증된 이미지를 고화질로 교체")
    parser.add_argument("lecture_id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    count = refresh(args.lecture_id, dry_run=args.dry_run)
    print(f"교체된 이미지 수: {count}")
