"""강의 요약에 실제로 도움이 되는 영상 장면만 선택한다.

- 타임라인을 기준으로 주변 프레임 최대 3개씩 후보로 만든다.
- 화질/중복을 1차 검사하고, 선택된 요약 API의 비전 모델로 2차 검사한다.
- 시각적인 관련성이 확인된 프레임만 MEDIA_ROOT에 저장한다.
- API 오류/판단 불가/부적절한 프레임이면 이미지 없이 진행한다(fail closed).
"""

import base64
import io
import json
import os
import re

import cv2
import numpy as np
from django.conf import settings
from PIL import Image

from ..utils.option_utils import normalize_summary_api
from .media_service import format_seconds_to_mmss
from .summary_service import (
    GEMINI_MODEL_NAME,
    OPENAI_FINAL_MODEL_NAME,
    _extract_openai_output_text,
    get_gemini_client,
    get_openai_client,
)

# 최대 4개 개념 × 3개 프레임 = 비전 API 1회 호출에 이미지 최대 12장.
IMAGE_CANDIDATE_OFFSETS_SECONDS = (-10, 5, 22)
IMAGE_MAX_CONCEPTS = 4
IMAGE_MAX_OUTPUTS = 4
IMAGE_JPEG_QUALITY = 72
IMAGE_FRAME_SIZE = (640, 360)

# off: 이미지를 삽입하지 않는다. 검증 없이 삽입하는 모드는 제공하지 않는다.
IMAGE_REVIEW_ENABLED = os.environ.get("EDUMATE_IMAGE_REVIEW", "on").lower() not in (
    "off", "false", "0", "no",
)


def _hash_frame(gray):
    """비슷한 화면을 여러 번 보내지 않기 위한 64비트 dHash."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _image_quality_is_usable(frame):
    """거의 빈 화면, 검은 화면, 흐릿한 전환 화면은 제외한다.

    PPT, 지도, 수식, 칠판 사진은 비전 검증에서 최종 판단하므로
    지나치게 높은 화질 임계값을 적용하지 않는다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 11.0:
        return False
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 14.0:
        return False
    edges = cv2.Canny(gray, 65, 130)
    if np.count_nonzero(edges) / edges.size < 0.002:
        return False
    return True


def _summary_context(summary_text, keyword):
    """핵심 개념이 서술된 주변 문장을 비전 검증의 참고 문맥으로 제공한다."""
    cleaned = re.sub(r"<[^>]+>", " ", summary_text or "")
    lines = [line.strip(" -*\t") for line in cleaned.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if keyword in line:
            nearby = " ".join(lines[index:index + 3])
            return nearby[:260]
    return (keyword or "")[:260]


def _collect_candidates(video_path, items, summary_text):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[이미지 관련성 검사] 영상 열기 실패: {video_path}")
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 and total_frames > 0 else None
    candidates = []

    try:
        for concept_index, item in enumerate(items[:IMAGE_MAX_CONCEPTS]):
            target = item.get("seconds")
            try:
                target = float(target)
            except (TypeError, ValueError):
                continue
            if target < 0 or (duration is not None and target >= duration):
                print(f"[이미지 관련성 검사] 범위 밖 타임라인 제외: {target}초")
                continue

            same_concept_hashes = []
            for offset_index, offset in enumerate(IMAGE_CANDIDATE_OFFSETS_SECONDS):
                second = target + offset
                # 영상 시작 전 오프셋을 0초로 당기면 강의 타이틀이 반복 후보가 된다.
                if second < 0 or (duration is not None and second >= duration):
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.resize(frame, IMAGE_FRAME_SIZE)
                if not _image_quality_is_usable(frame):
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fingerprint = _hash_frame(gray)
                if any((fingerprint ^ old).bit_count() <= 5 for old in same_concept_hashes):
                    continue
                same_concept_hashes.append(fingerprint)

                encoded_ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_JPEG_QUALITY]
                )
                if not encoded_ok:
                    continue

                candidates.append({
                    "id": f"{concept_index}_{offset_index}",
                    "concept_index": concept_index,
                    "keyword": item.get("keyword", ""),
                    "target_seconds": int(round(target)),
                    "capture_seconds": int(round(second)),
                    "context": _summary_context(summary_text, item.get("keyword", "")),
                    "jpeg": encoded.tobytes(),
                    "fingerprint": fingerprint,
                })
    finally:
        cap.release()

    return candidates


def _review_prompt(candidates):
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["concept_index"], {
            "keyword": candidate["keyword"],
            "context": candidate["context"],
            "ids": [],
        })["ids"].append(candidate["id"])

    concept_lines = []
    for group_index, group in groups.items():
        concept_lines.append(
            f"Group {group_index}: concept={group['keyword']!r}; "
            f"summary_context={group['context']!r}; "
            f"candidate_ids={', '.join(group['ids'])}"
        )

    return (
        "You are a strict visual evidence reviewer for educational lecture notes. "
        "Images follow this instruction, each with a candidate ID and video timestamp.\n"
        "For EACH concept group, select ZERO OR ONE frame only when its VISIBLE content "
        "directly helps understand that specific concept in the summary context. "
        "Accept readable relevant slides, diagrams, equations, code, maps, charts, "
        "demonstrations, or informative board writing. Reject unrelated slides, "
        "presenter/talking-head-only frames, logos, ads, transition screens, "
        "generic scenery, duplicated generic visuals, and visually unclear content. "
        "A timestamp, a spoken topic, or a caption alone is NOT visual evidence of relevance. "
        "Do not infer relevance from the accompanying label if the image itself does not show it. "
        "When unsure, select nothing. Never fill a quota.\n"
        "Return ONLY compact JSON: {\"selected_ids\":[\"0_1\",\"1_0\"]}. "
        "Use exact supplied IDs. [] is valid and preferred to uncertain choices.\n\n"
        + "\n".join(concept_lines)
    )


def _extract_selected_ids(raw_text, candidates):
    """AI가 반환한 값을 허용된 ID 목록으로 제한한다."""
    text = (raw_text or "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return []
    ids = payload.get("selected_ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list):
        return []

    by_id = {c["id"]: c for c in candidates}
    result = []
    used_groups = set()
    for candidate_id in ids:
        if not isinstance(candidate_id, str) or candidate_id not in by_id:
            continue
        candidate = by_id[candidate_id]
        if candidate["concept_index"] in used_groups:
            continue
        used_groups.add(candidate["concept_index"])
        result.append(candidate)
    return result


def _review_with_openai(candidates):
    content = [{"type": "input_text", "text": _review_prompt(candidates)}]
    for candidate in candidates:
        content.append({
            "type": "input_text",
            "text": (
                f"Candidate ID {candidate['id']}, "
                f"video time {format_seconds_to_mmss(candidate['capture_seconds'])}"
            ),
        })
        content.append({
            "type": "input_image",
            "image_url": "data:image/jpeg;base64," + base64.b64encode(candidate["jpeg"]).decode("ascii"),
            "detail": "low",
        })

    response = get_openai_client().responses.create(
        model=os.environ.get("EDUMATE_IMAGE_REVIEW_GPT_MODEL", OPENAI_FINAL_MODEL_NAME),
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "none"},
        max_output_tokens=350,
    )
    return _extract_openai_output_text(response)


def _review_with_gemini(candidates):
    contents = [_review_prompt(candidates)]
    for candidate in candidates:
        contents.append(
            f"Candidate ID {candidate['id']}, "
            f"video time {format_seconds_to_mmss(candidate['capture_seconds'])}"
        )
        with Image.open(io.BytesIO(candidate["jpeg"])) as image:
            contents.append(image.copy())
    response = get_gemini_client().models.generate_content(
        model=os.environ.get("EDUMATE_IMAGE_REVIEW_GEMINI_MODEL", GEMINI_MODEL_NAME),
        contents=contents,
    )
    return getattr(response, "text", "") or ""


def select_relevant_timeline_frames(
    video_path, lecture_id, timeline_items, summary_text,
    summary_api="gpt", max_images=IMAGE_MAX_OUTPUTS,
):
    """실제 개념과 관련된 시각 자료만 승인하고 보관한다.

    오류가 발생해도 상위 강의 분석을 실패시키지 않는다.
    반환 프레임에는 삽입할 개념의 target_seconds가 함께 들어있다.
    """
    if not IMAGE_REVIEW_ENABLED:
        print("[이미지 관련성 검사] EDUMATE_IMAGE_REVIEW=off: 이미지 없이 진행")
        return []
    if not video_path or not os.path.isfile(video_path) or not timeline_items:
        return []

    try:
        candidates = _collect_candidates(video_path, timeline_items, summary_text)
        print(f"[이미지 관련성 검사] 검토 후보 {len(candidates)}장")
        if not candidates:
            return []

        if normalize_summary_api(summary_api) == "gpt":
            raw_decision = _review_with_openai(candidates)
        else:
            raw_decision = _review_with_gemini(candidates)

        approved = _extract_selected_ids(raw_decision, candidates)
        print(f"[이미지 관련성 검사] AI 승인 {len(approved)}장")
        if not approved:
            return []

        output_dir = os.path.join(settings.MEDIA_ROOT, "lecture_frames", str(lecture_id))
        os.makedirs(output_dir, exist_ok=True)
        frames = []
        used_hashes = []
        for candidate in approved:
            if len(frames) >= max_images:
                break
            fingerprint = candidate["fingerprint"]
            if any((fingerprint ^ old).bit_count() <= 5 for old in used_hashes):
                continue
            used_hashes.append(fingerprint)
            image_time = format_seconds_to_mmss(candidate["capture_seconds"])
            safe_time = image_time.replace(":", "_")
            filename = f"verified_{candidate['concept_index']}_{safe_time}.jpg"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as output:
                output.write(candidate["jpeg"])

            frames.append({
                "time_str": image_time,
                "filepath": filepath,
                "seconds": candidate["capture_seconds"],
                "target_seconds": candidate["target_seconds"],
                "keyword": candidate["keyword"],
                "validated": True,
                "capture_type": "vision_verified",
            })
        print(f"[이미지 관련성 검사] 최종 저장 {len(frames)}장")
        return frames
    except Exception as exc:
        print(f"[이미지 관련성 검사] 실패, 이미지 없이 진행: {exc}")
        return []
