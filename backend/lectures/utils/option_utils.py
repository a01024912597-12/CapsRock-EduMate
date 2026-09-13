# 업로드/분석 옵션 정규화 유틸

WHISPER_MODEL_NAME = "base"
PARALLEL_MAX_WORKERS = 2
SUMMARY_API_NAME = "gemini"
SUBJECT_CODE_NAME = "auto"


def normalize_whisper_model_name(value):
    """업로드 페이지에서 전달된 Whisper 모델명을 안전하게 정리한다."""
    value = (value or WHISPER_MODEL_NAME).strip().lower()

    allowed_models = ["base", "small", "medium", "large"]

    if value not in allowed_models:
        return WHISPER_MODEL_NAME

    return value


def normalize_subject_code(value):
    """업로드 페이지에서 전달된 과목 코드를 안전하게 정리한다.

    auto: AI 자동 판단 및 시험 대비 요약
    1   : 역사/인문학
    2   : 프로그래밍/IT
    3   : 수학/과학
    4   : 일반 강의
    """
    value = str(value or SUBJECT_CODE_NAME).strip().lower()

    allowed_codes = ["auto", "1", "2", "3", "4"]

    if value not in allowed_codes:
        return SUBJECT_CODE_NAME

    return value


def normalize_worker_count(value):
    """업로드 페이지에서 전달된 workers 값을 1~4 범위로 정리한다."""
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return PARALLEL_MAX_WORKERS

    if workers < 1:
        return 1

    if workers > 4:
        return 4

    return workers


def normalize_summary_api(value):
    """업로드 페이지에서 전달된 요약 API 값을 안전하게 정리한다.

    gemini: 기존 Gemini API 요약
    gpt   : OpenAI GPT API 요약
    """
    value = str(value or SUMMARY_API_NAME).strip().lower()

    allowed_values = ["gemini", "gpt"]

    if value not in allowed_values:
        return SUMMARY_API_NAME

    return value


def get_lecture_analysis_options(request, lecture_id):
    """강의별 분석 옵션을 세션에서 가져온다."""
    session_key = f"lecture_analysis_options_{lecture_id}"

    options = request.session.get(session_key, {})

    return {
        "whisper_model": normalize_whisper_model_name(
            options.get("whisper_model", WHISPER_MODEL_NAME)
        ),
        "workers": normalize_worker_count(
            options.get("workers", PARALLEL_MAX_WORKERS)
        ),
        "subject_code": normalize_subject_code(
            options.get("subject_code", SUBJECT_CODE_NAME)
        ),
        "summary_api": normalize_summary_api(
            options.get("summary_api", SUMMARY_API_NAME)
        ),
    }
