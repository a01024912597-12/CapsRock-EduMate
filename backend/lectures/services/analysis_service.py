# C:\CapsRock-EduMate\backend\lectures\services\analysis_service.py

import os
import time
import concurrent.futures

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone

from ..models import Lecture
from ..utils.option_utils import get_lecture_analysis_options
from . import stt_service
from .stt_service import (
    AUDIO_SEGMENT_SECONDS,
    get_audio_segments_for_lecture,
    lecture_has_local_video,
    process_segment_task,
    cleanup_audio_segments,
    chunks_from_whisper_segments,
)
from .media_service import (
    download_low_res_video,
    capture_frame_at_timestamp,
)
from .summary_service import (
    get_summary_model_label,
    summarize_chunk,
    make_final_summary,
    replace_image_markers_with_html,
    extract_summary_timeline,
    parse_summary_timeline_items,
    inject_core_timeline_images,
)


# =========================
# 요약 속도 최적화 설정
# =========================
# 값이 클수록 chunk 개수가 줄어 GPT/Gemini 호출 횟수가 줄어든다.
SUMMARY_CHUNK_SEGMENTS = 24
SUMMARY_CHUNK_MAX_CHARS = 8000

# chunk 요약 API 병렬 처리 수.
# 429, RESOURCE_EXHAUSTED, 503 에러가 자주 뜨면 1로 낮춘다.
SUMMARY_API_PARALLEL_WORKERS = 2

# 최종 요약 모델에는 이미지를 넘기지 않고,
# 요약 완료 후 핵심 타임라인 기준으로만 이미지를 직접 캡처한다.
SUMMARY_PASS_IMAGES_TO_FINAL_MODEL = False


# 분석 실패 시 저장하는 공통 필드
ANALYSIS_FAILURE_UPDATE_FIELDS = [
    "analysis_duration_seconds",
    "stt_duration_seconds",
    "summary_duration_seconds",
    "whisper_model_name",
    "analyzed_at",
]


def _save_analysis_failure(lecture, analysis_duration, stt_duration, whisper_model_name):
    """분석 실패 상태를 DB에 저장한다."""
    lecture.analysis_duration_seconds = analysis_duration
    lecture.stt_duration_seconds = stt_duration
    lecture.summary_duration_seconds = 0
    lecture.whisper_model_name = whisper_model_name
    lecture.analyzed_at = timezone.now()
    lecture.save(update_fields=ANALYSIS_FAILURE_UPDATE_FIELDS)


def analyze_video_request(request):
    """강의 분석 API.

    흐름:
    0. 업로드 페이지에서 선택한 Whisper 모델과 workers 수를 세션에서 불러온다.
    1. 유튜브 또는 업로드 영상에서 오디오를 추출하고 FFmpeg로 180초 단위 분할한다.
    2. 선택된 workers 수에 맞춰 Whisper STT를 병렬 처리한다.
    3. Whisper 세그먼트 기반 타임라인 청크를 생성한다.
    4. 업로드 페이지에서 선택한 요약 API로 부분 요약 및 최종 요약을 수행한다.
    5. 요약문 핵심 타임라인을 기준으로 필요한 이미지만 직접 캡처한다.
    6. 요약 결과와 분석 시간을 DB에 저장한다.
    """
    lecture_id = request.GET.get("lecture_id")

    if not lecture_id:
        return JsonResponse({
            "status": "error",
            "message": "lecture_id가 전달되지 않았습니다.",
        })

    lecture = get_object_or_404(Lecture, id=lecture_id, user=request.user)

    # 업로드 페이지에서 저장한 분석 옵션 불러오기
    analysis_options = get_lecture_analysis_options(request, lecture.id)
    selected_whisper_model = analysis_options["whisper_model"]
    selected_workers = analysis_options["workers"]
    selected_subject_code = analysis_options["subject_code"]
    selected_summary_api = analysis_options.get("summary_api", "gemini")
    summary_model_label = get_summary_model_label(selected_summary_api)

    # get_whisper_model()이 참조하는 stt_service의 모델명을 현재 강의 설정값으로 변경
    stt_service.WHISPER_MODEL_NAME = selected_whisper_model

    # 이미 요약이 존재하면 기존 결과 반환
    if lecture.summary_text:
        summary_timeline = extract_summary_timeline(lecture.summary_text)

        if (
            hasattr(lecture, "summary_timeline")
            and summary_timeline
            and lecture.summary_timeline != summary_timeline
        ):
            lecture.summary_timeline = summary_timeline
            lecture.save(update_fields=["summary_timeline"])

        return JsonResponse({
            "status": "success",
            "result": lecture.summary_text,
            "summary_timeline": getattr(lecture, "summary_timeline", summary_timeline),
            "analysis_duration_seconds": lecture.analysis_duration_seconds,
            "stt_duration_seconds": lecture.stt_duration_seconds,
            "summary_duration_seconds": lecture.summary_duration_seconds,
            "whisper_model_name": lecture.whisper_model_name,
            "selected_whisper_model": selected_whisper_model,
            "selected_workers": selected_workers,
            "selected_summary_api": selected_summary_api,
        })

    print("\n" + "=" * 60)
    print(f"[병렬 STT + 선택 API 멀티모달 요약 시작] 강의 제목: {lecture.title}")
    print(
        f"[설정] Whisper={selected_whisper_model}, "
        f"workers={selected_workers}, "
        f"subject_code={selected_subject_code}, "
        f"summary_api={selected_summary_api}, "
        f"segment={AUDIO_SEGMENT_SECONDS}초, "
        f"summary_model={summary_model_label}"
    )
    print("=" * 60)

    analysis_start_time = time.time()
    segments = []
    timeline_frames = []
    temp_video_path = os.path.join(settings.BASE_DIR, "temp_video.mp4")

    try:
        # =========================
        # 0. 핵심 이미지 캡처용 영상 경로 준비
        # =========================
        print("0. 핵심 이미지 캡처용 영상 경로 준비 중...")

        frame_video_path = ""
        should_remove_temp_video = False

        if lecture_has_local_video(lecture):
            frame_video_path = lecture.video_file.path
            print(f"[핵심 이미지 캡처] 업로드 영상 사용: {frame_video_path}")
        else:
            print("[핵심 이미지 캡처] 유튜브 영상은 요약 완료 후 필요한 경우에만 다운로드합니다.")

        # =========================
        # 1. 오디오 추출 및 분할
        # =========================
        print("1. 오디오 추출 및 구간 분할 중...")
        segments = get_audio_segments_for_lecture(lecture)

        if not segments:
            analysis_duration = time.time() - analysis_start_time
            _save_analysis_failure(
                lecture=lecture,
                analysis_duration=analysis_duration,
                stt_duration=0,
                whisper_model_name=(
                    f"{selected_whisper_model} / "
                    f"audio_segment_failed / "
                    f"workers_{selected_workers}"
                ),
            )

            return JsonResponse({
                "status": "error",
                "message": "오디오 추출 또는 분할에 실패했습니다.",
            })

        # =========================
        # 2. 병렬 Whisper STT
        # =========================
        print(f"2. 병렬 Whisper STT 시작: 총 {len(segments)}개 구간")

        segment_tasks = []

        for index, segment_path in enumerate(segments):
            offset_seconds = index * AUDIO_SEGMENT_SECONDS
            segment_tasks.append((segment_path, offset_seconds))

        with concurrent.futures.ThreadPoolExecutor(max_workers=selected_workers) as executor:
            results = list(executor.map(process_segment_task, segment_tasks))

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

                try:
                    segment_index = segments.index(segment_path)
                except ValueError:
                    segment_index = 0

                offset_seconds = segment_index * AUDIO_SEGMENT_SECONDS
                retry_result = process_segment_task((segment_path, offset_seconds))

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
        all_whisper_segments = []
        total_stt_duration = 0.0
        final_failed_segments = []

        for index, result in enumerate(results, start=1):
            text = result.get("text", "").strip()
            total_stt_duration += result.get("stt_duration", 0)

            if text:
                transcript_parts.append(f"[{index}구간 전사]\n{text}")

            if result.get("segments"):
                all_whisper_segments.extend(result.get("segments") or [])

            if not text and not result.get("segments"):
                final_failed_segments.append(result.get("segment", f"{index}구간"))

        # STT 처리 후 chunk 파일 정리
        cleanup_audio_segments(segments)

        if final_failed_segments:
            analysis_duration = time.time() - analysis_start_time
            _save_analysis_failure(
                lecture=lecture,
                analysis_duration=analysis_duration,
                stt_duration=total_stt_duration,
                whisper_model_name=(
                    f"{selected_whisper_model} / "
                    f"failed_segments / "
                    f"workers_{selected_workers}"
                ),
            )

            failed_names = ", ".join(final_failed_segments)

            print(f"--- [최종 실패] STT 실패 구간: {failed_names} ---")

            return JsonResponse({
                "status": "error",
                "message": f"일부 구간 STT에 실패했습니다: {failed_names}",
            })

        full_transcript = "\n\n".join(transcript_parts).strip()

        if not full_transcript and not all_whisper_segments:
            analysis_duration = time.time() - analysis_start_time
            _save_analysis_failure(
                lecture=lecture,
                analysis_duration=analysis_duration,
                stt_duration=total_stt_duration,
                whisper_model_name=(
                    f"{selected_whisper_model} / "
                    f"parallel_stt_empty / "
                    f"workers_{selected_workers}"
                ),
            )

            return JsonResponse({
                "status": "error",
                "message": "STT 결과가 비어 있습니다.",
            })

        print("3. 병렬 STT 완료")
        print(f"--- [디버깅] 전체 전사 길이: {len(full_transcript)}")
        print(f"--- [디버깅] 전체 전사 앞부분: {full_transcript[:300]}")

        # =========================
        # 3. 타임라인 기반 요약 청크 생성
        # =========================
        print(f"4. {summary_model_label} 요약 시작")
        summary_start_time = time.time()

        timeline_chunks = chunks_from_whisper_segments(
            all_whisper_segments,
            segments_per_chunk=SUMMARY_CHUNK_SEGMENTS,
            max_chars_per_chunk=SUMMARY_CHUNK_MAX_CHARS,
        )

        if timeline_chunks:
            print(
                f"--- [{summary_model_label} 요약 방식] Whisper 세그먼트 기반 "
                f"타임라인 청크 요약: {len(timeline_chunks)}개 ---"
            )

            def summarize_chunk_task(index_chunk):
                index, chunk = index_chunk
                print(f"--- [{summary_model_label} chunk 요약] {index}/{len(timeline_chunks)} ---")
                return summarize_chunk(
                    chunk,
                    subject_code=selected_subject_code,
                    summary_api=selected_summary_api,
                )

            chunk_summary_tasks = list(enumerate(timeline_chunks, start=1))
            chunk_summary_workers = min(
                SUMMARY_API_PARALLEL_WORKERS,
                len(chunk_summary_tasks),
            )

            if chunk_summary_workers <= 1:
                chunk_summaries = [
                    summarize_chunk_task(task)
                    for task in chunk_summary_tasks
                ]
            else:
                print(
                    f"--- [{summary_model_label} chunk 요약 병렬 처리] "
                    f"workers={chunk_summary_workers} ---"
                )

                with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_summary_workers) as executor:
                    chunk_summaries = list(
                        executor.map(summarize_chunk_task, chunk_summary_tasks)
                    )

            merged_chunk_summaries = "\n\n".join(chunk_summaries)

        else:
            print(f"--- [{summary_model_label} 요약 방식] 세그먼트 없음, 전체 전사문 기반 요약 ---")
            merged_chunk_summaries = full_transcript

        # =========================
        # 4. 멀티모달 최종 요약
        # =========================
        final_summary = make_final_summary(
            merged_chunk_summaries,
            timeline_frames=[] if not SUMMARY_PASS_IMAGES_TO_FINAL_MODEL else timeline_frames,
            subject_code=selected_subject_code,
            summary_api=selected_summary_api,
        )

        final_summary = replace_image_markers_with_html(
            ai_summary_text=final_summary,
            lecture_id=lecture.id,
        )

        # =========================
        # 5. 핵심 개념 타임라인 기반 이미지 자동 삽입
        # =========================
        core_timeline_items = parse_summary_timeline_items(
            final_summary,
            max_items=4,
        )

        if core_timeline_items:
            print(
                f"[핵심 이미지 삽입] 요약 핵심 타임라인 "
                f"{len(core_timeline_items)}개 감지"
            )

        if core_timeline_items and not frame_video_path and not lecture_has_local_video(lecture):
            print("[핵심 이미지 캡처] 유튜브 저화질 영상 다운로드 시작...")

            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except OSError:
                pass

            video_downloaded = download_low_res_video(
                lecture.youtube_url,
                temp_video_path,
            )

            if video_downloaded:
                frame_video_path = temp_video_path
                should_remove_temp_video = True
            else:
                print("[핵심 이미지 캡처] 유튜브 영상 다운로드 실패, 이미지 없이 요약을 저장합니다.")

        core_timeline_frames = []

        if frame_video_path and os.path.exists(frame_video_path):
            for item in core_timeline_items:
                captured_frame = capture_frame_at_timestamp(
                    video_path=frame_video_path,
                    lecture_id=lecture.id,
                    seconds=item.get("seconds"),
                    keyword=item.get("keyword", ""),
                    output_prefix="core",
                )

                if captured_frame:
                    core_timeline_frames.append(captured_frame)

        timeline_frames = core_timeline_frames

        final_summary = inject_core_timeline_images(
            summary_text=final_summary,
            timeline_frames=timeline_frames,
            lecture_id=lecture.id,
            max_images=4,
        )

        if should_remove_temp_video:
            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except OSError:
                pass

        summary_duration = time.time() - summary_start_time

        result_text = "[AI가 분석한 강의 요약]\n\n" + final_summary
        summary_timeline = extract_summary_timeline(result_text)

        analysis_duration = time.time() - analysis_start_time

        lecture.summary_text = result_text
        lecture.analysis_duration_seconds = analysis_duration
        lecture.stt_duration_seconds = total_stt_duration
        lecture.summary_duration_seconds = summary_duration
        lecture.whisper_model_name = (
            f"{selected_whisper_model} / "
            f"parallel_stt_workers_{selected_workers}_retry / "
            f"summary_{selected_summary_api}_{summary_model_label}"
        )
        lecture.analyzed_at = timezone.now()

        update_fields = [
            "summary_text",
            "analysis_duration_seconds",
            "stt_duration_seconds",
            "summary_duration_seconds",
            "whisper_model_name",
            "analyzed_at",
        ]

        if hasattr(lecture, "summary_timeline"):
            lecture.summary_timeline = summary_timeline
            update_fields.append("summary_timeline")

        lecture.save(update_fields=update_fields)

        print("=" * 60)
        print("[병렬 STT + 선택 API 멀티모달 요약 완료]")
        print(f"전체 분석 시간: {analysis_duration:.2f}초")
        print(f"구간별 STT 시간 합계: {total_stt_duration:.2f}초")
        print(f"{summary_model_label} 요약 시간: {summary_duration:.2f}초")
        print(f"추출 이미지 프레임 수: {len(timeline_frames)}")
        print(f"whisper_model: {selected_whisper_model}")
        print(f"workers: {selected_workers}")
        print(f"summary_api: {selected_summary_api}")
        print("=" * 60 + "\n")

        return JsonResponse({
            "status": "success",
            "result": result_text,
            "summary_timeline": summary_timeline,
            "analysis_duration_seconds": analysis_duration,
            "stt_duration_seconds": total_stt_duration,
            "summary_duration_seconds": summary_duration,
            "whisper_model_name": lecture.whisper_model_name,
            "selected_whisper_model": selected_whisper_model,
            "selected_workers": selected_workers,
            "selected_summary_api": selected_summary_api,
            "timeline_frame_count": len(timeline_frames),
        })

    except Exception as e:
        cleanup_audio_segments(segments)

        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except OSError:
            pass

        analysis_duration = time.time() - analysis_start_time
        _save_analysis_failure(
            lecture=lecture,
            analysis_duration=analysis_duration,
            stt_duration=0,
            whisper_model_name=(
                f"{selected_whisper_model} / "
                f"parallel_error / "
                f"workers_{selected_workers} / "
                f"summary_{selected_summary_api}_{summary_model_label}"
            ),
        )

        print(f"[병렬 STT + 선택 API 멀티모달 요약 시스템 에러] {str(e)}")

        return JsonResponse({
            "status": "error",
            "message": str(e),
        })