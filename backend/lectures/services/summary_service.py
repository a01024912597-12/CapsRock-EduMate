import os
import re
import time
import base64
import mimetypes

import PIL.Image
from google import genai
from django.conf import settings

from ..utils.option_utils import normalize_subject_code, normalize_summary_api
from ..utils.text_utils import clean_summary_text
from .media_service import format_seconds_to_mmss


# =========================
# Gemini API 설정
# =========================

GEMINI_MODEL_NAME = getattr(settings, "GEMINI_MODEL_NAME", "gemini-3.8-flash")
_gemini_client = None

# =========================
# OpenAI GPT API 설정
# =========================
# OpenAI GPT API 설정
# =========================

# 로컬 테스트 편의를 위해 하드코딩 방식으로 둔다.
# 실제 GitHub 업로드/팀원 공유 전에는 반드시 키를 제거하거나 환경변수 방식으로 바꿔야 한다.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# GPT 모델 설정
# 청크 요약: 비용 절감을 위해 Terra 사용
OPENAI_CHUNK_MODEL_NAME = "gpt-5.6-terra"

# 최종 요약: 품질 유지를 위해 Sol 사용
OPENAI_FINAL_MODEL_NAME = "gpt-5.6-sol"

# GPT 최종 요약 성능 테스트 설정
OPENAI_FINAL_REASONING_EFFORT = "none"
OPENAI_FINAL_MAX_OUTPUT_TOKENS = 7000


# 최종 요약 분량 자동 조정 설정
# STT 원문 글자 수를 기준으로 권장 요약 분량을 계산한다.
ADAPTIVE_SUMMARY_RATIO = 0.10
ADAPTIVE_SUMMARY_MIN_CHARS = 1000
ADAPTIVE_SUMMARY_MAX_CHARS = 7000
ADAPTIVE_SUMMARY_MAX_RANGE_CHARS = 8000


_openai_client = None

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


def gemini_generate_text(contents, max_retries=5, base_delay=10):
    """Gemini API를 호출하여 텍스트 응답을 반환한다.

    문자열 prompt와 이미지가 포함된 contents 리스트를 모두 받을 수 있다.
    503, 429, RESOURCE_EXHAUSTED 같은 일시적 오류는 재시도한다.
    단, 선불 크레딧 소진 오류는 재시도하지 않고 바로 중단한다.
    """
    client = get_gemini_client()
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=contents,
            )

            text = getattr(response, "text", None) or ""

            if text:
                return text.strip()

            print(f"[Gemini 경고] 응답 text가 비어 있습니다. attempt={attempt}")
            return ""

        except Exception as e:
            last_error = e
            error_text = str(e)

            print(f"[Gemini 호출 에러] attempt={attempt}/{max_retries} / {error_text}")

            if "prepayment credits are depleted" in error_text:
                raise RuntimeError(
                    "Gemini API 선불 크레딧이 소진되었습니다. "
                    "Google AI Studio 또는 Google Cloud에서 결제/크레딧 상태를 확인하세요."
                )

            is_retryable = any(
                token in error_text
                for token in (
                    "503",
                    "429",
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED",
                    "high demand",
                    "temporarily",
                )
            )

            if is_retryable and attempt < max_retries:
                wait_seconds = base_delay * (2 ** (attempt - 1))
                print(f"[Gemini 재시도 대기] {wait_seconds}초 후 재시도합니다.")
                time.sleep(wait_seconds)
                continue

            raise

    print(f"[Gemini 최종 실패] 모든 재시도 실패: {last_error}")
    raise last_error


# =========================
# 요약 API 선택 공통 처리
# =========================

def get_summary_model_label(summary_api="gemini"):
    """로그 및 DB 기록용 요약 API/모델명을 반환한다."""
    summary_api = normalize_summary_api(summary_api)

    if summary_api == "gpt":
        return (
            f"GPT/chunk={OPENAI_CHUNK_MODEL_NAME}, "
            f"final={OPENAI_FINAL_MODEL_NAME}"
        )

    return f"Gemini/{GEMINI_MODEL_NAME}"


def get_openai_client():
    """OpenAI GPT API 클라이언트를 지연 생성한다."""
    global _openai_client

    if _openai_client is not None:
        return _openai_client

    api_key = (OPENAI_API_KEY or "").strip()

    if not api_key or api_key == "여기에_OPENAI_API_KEY_입력":
        raise ValueError(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "lectures/services/summary_service.py 파일의 OPENAI_API_KEY 값에 실제 키를 입력하세요."
        )

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai 패키지가 설치되지 않았습니다. "
            "가상환경에서 python -m pip install openai 명령을 실행하세요."
        ) from e

    _openai_client = OpenAI(
        api_key=api_key,
        timeout=180.0,
        max_retries=2,
    )

    return _openai_client


def _extract_openai_output_text(response):
    """OpenAI Responses API 응답에서 텍스트만 안전하게 추출한다."""
    text = getattr(response, "output_text", "")

    if text:
        return text.strip()

    collected = []

    for output_item in getattr(response, "output", []) or []:
        for content in getattr(output_item, "content", []) or []:
            content_text = getattr(content, "text", "")

            if content_text:
                collected.append(content_text)

    return "\n".join(collected).strip()

def openai_generate_text(prompt):
    """OpenAI GPT API를 호출하여 텍스트 응답을 반환한다."""
    client = get_openai_client()

    print(
        f"[GPT 청크 요약 설정] "
        f"model={OPENAI_CHUNK_MODEL_NAME}, "
        f"reasoning=none"
    )

    retry_wait_seconds = [5, 15, 30]
    last_error = None

    for attempt, wait_seconds in enumerate(retry_wait_seconds, start=1):
        try:
            response = client.responses.create(
                model=OPENAI_CHUNK_MODEL_NAME,
                input=prompt,
                reasoning={
                    "effort": "none",
                },
                max_output_tokens=5000,
            )

            text = _extract_openai_output_text(response)

            if text:
                return text.strip()

            print(f"[GPT 경고] 응답 text가 비어 있습니다. attempt={attempt}")
            return ""

        except Exception as e:
            last_error = e
            error_text = str(e)

            print(f"[GPT 호출 에러] attempt={attempt} / {error_text}")

            is_retryable = (
                "429" in error_text
                or "rate limit" in error_text.lower()
                or "500" in error_text
                or "502" in error_text
                or "503" in error_text
                or "504" in error_text
                or "timeout" in error_text.lower()
                or "temporarily" in error_text.lower()
            )

            if is_retryable:
                print(f"[GPT 재시도 대기] {wait_seconds}초 후 재시도합니다.")
                time.sleep(wait_seconds)
                continue

            raise e

    print(f"[GPT 최종 실패] 모든 재시도 실패: {last_error}")
    raise last_error


def _image_file_to_data_url(filepath):
    """로컬 이미지 파일을 OpenAI 이미지 입력용 data URL로 변환한다."""
    mime_type, _ = mimetypes.guess_type(filepath)

    if not mime_type:
        mime_type = "image/jpeg"

    with open(filepath, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def build_adaptive_summary_plan(source_char_count):
    """STT 원문 길이를 기준으로 최종 요약 권장 분량과 구조를 계산한다.

    영상 길이가 아니라 실제 전사 텍스트의 양을 기준으로 하므로,
    말이 적은 긴 영상과 말이 많은 짧은 영상을 더 자연스럽게 구분할 수 있다.
    """
    try:
        source_chars = max(int(source_char_count or 0), 0)
    except (TypeError, ValueError):
        source_chars = 0

    target_chars = int(source_chars * ADAPTIVE_SUMMARY_RATIO)
    target_chars = max(ADAPTIVE_SUMMARY_MIN_CHARS, target_chars)
    target_chars = min(ADAPTIVE_SUMMARY_MAX_CHARS, target_chars)

    target_min_chars = max(800, int(target_chars * 0.85))
    target_max_chars = min(
        ADAPTIVE_SUMMARY_MAX_RANGE_CHARS,
        max(target_min_chars + 200, int(target_chars * 1.15)),
    )

    if source_chars <= 8000:
        overview_paragraph_max = 1
        concept_max = 3
    elif source_chars <= 15000:
        overview_paragraph_max = 2
        concept_max = 4
    elif source_chars <= 30000:
        overview_paragraph_max = 3
        concept_max = 6
    elif source_chars <= 60000:
        overview_paragraph_max = 4
        concept_max = 8
    else:
        overview_paragraph_max = 5
        concept_max = 10

    return {
        "source_chars": source_chars,
        "target_chars": target_chars,
        "target_min_chars": target_min_chars,
        "target_max_chars": target_max_chars,
        "overview_paragraph_max": overview_paragraph_max,
        "concept_max": concept_max,
    }


def _get_openai_usage_value(usage, field_name):
    """OpenAI usage 객체에서 토큰 값을 안전하게 읽는다."""
    if usage is None:
        return None

    if isinstance(usage, dict):
        return usage.get(field_name)

    return getattr(usage, field_name, None)


def _log_openai_response_metadata(response, label="GPT 응답"):
    """Responses API의 완료 여부와 실제 토큰 사용량을 로그에 남긴다."""
    status = getattr(response, "status", None) or "unknown"
    incomplete_details = getattr(response, "incomplete_details", None)

    if isinstance(incomplete_details, dict):
        incomplete_reason = incomplete_details.get("reason")
    else:
        incomplete_reason = getattr(incomplete_details, "reason", None)

    usage = getattr(response, "usage", None)
    input_tokens = _get_openai_usage_value(usage, "input_tokens")
    output_tokens = _get_openai_usage_value(usage, "output_tokens")
    total_tokens = _get_openai_usage_value(usage, "total_tokens")

    print(
        f"[{label}] status={status}, "
        f"incomplete_reason={incomplete_reason or '없음'}, "
        f"input_tokens={input_tokens if input_tokens is not None else '확인불가'}, "
        f"output_tokens={output_tokens if output_tokens is not None else '확인불가'}, "
        f"total_tokens={total_tokens if total_tokens is not None else '확인불가'}"
    )

    return {
        "status": status,
        "incomplete_reason": incomplete_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def openai_generate_multimodal_text(
    prompt_text,
    timeline_frames=None,
    target_max_chars=None,
):
    """OpenAI GPT API를 이용해 텍스트와 이미지 프레임을 함께 요약한다."""
    client = get_openai_client()

    content = [
        {
            "type": "input_text",
            "text": prompt_text,
        }
    ]

    if timeline_frames:
        for frame in timeline_frames:
            filepath = frame.get("filepath")

            if not filepath or not os.path.exists(filepath):
                continue

            try:
                content.append({
                    "type": "input_text",
                    "text": f"[{frame.get('time_str', '')} 시점의 칠판/PPT 화면]",
                })
                content.append({
                    "type": "input_image",
                    "image_url": _image_file_to_data_url(filepath),
                })
            except Exception as e:
                print(f"[GPT 요약 엔지니어링] 이미지 첨부 실패: {e}")

    retry_wait_seconds = [5, 15, 30]
    last_error = None

    for attempt, wait_seconds in enumerate(retry_wait_seconds, start=1):
        try:
            print(
                f"[GPT 최종 요약 설정] "
                f"model={OPENAI_FINAL_MODEL_NAME}, "
                f"reasoning={OPENAI_FINAL_REASONING_EFFORT}, "
                f"max_output_tokens={OPENAI_FINAL_MAX_OUTPUT_TOKENS}"
            )

            response = client.responses.create(
                model=OPENAI_FINAL_MODEL_NAME,
                input=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                reasoning={
                    "effort": OPENAI_FINAL_REASONING_EFFORT,
                },
                max_output_tokens=OPENAI_FINAL_MAX_OUTPUT_TOKENS,
            )
            response_meta = _log_openai_response_metadata(
                response,
                label="GPT 최종 요약 응답",
            )
            text = _extract_openai_output_text(response)

            is_incomplete = (
                response_meta.get("status") == "incomplete"
                or bool(response_meta.get("incomplete_reason"))
            )

            if text and not is_incomplete:
                print(f"[GPT 최종 요약 길이] {len(text)}자")
                return text.strip()

            if text and is_incomplete:
                # 토큰 제한 등으로 문장이 중간에서 잘렸다면, 같은 원문으로 한 번만
                # 더 짧고 완결된 결과를 재생성한다. 평소에는 추가 호출이 발생하지 않는다.
                safe_max_chars = target_max_chars or ADAPTIVE_SUMMARY_MAX_RANGE_CHARS
                print(
                    f"[GPT 최종 요약 경고] 응답이 완결되지 않았습니다. "
                    f"reason={response_meta.get('incomplete_reason') or response_meta.get('status')} / "
                    f"{safe_max_chars}자 이내 완성본으로 1회 재생성합니다."
                )

                retry_prompt = (
                    prompt_text
                    + "\n\n[길이 초과 재생성 필수 규칙]\n"
                    + f"- 이전 생성은 출력 제한으로 완결되지 않았다. 이번에는 반드시 {safe_max_chars:,}자 이내에서 끝낼 것.\n"
                    + "- 기존 내용을 그대로 반복해 분량을 늘리지 말 것.\n"
                    + "- 중요한 개념, 예시, 수치, 인과관계는 보존하되 중복 표현을 압축할 것.\n"
                    + "- 1~4번 섹션을 모두 완성하고 마지막 문장을 완결한 뒤 종료할 것.\n"
                )

                retry_content = [
                    {
                        "type": "input_text",
                        "text": retry_prompt,
                    }
                ]

                if timeline_frames:
                    for frame in timeline_frames:
                        filepath = frame.get("filepath")
                        if not filepath or not os.path.exists(filepath):
                            continue

                        try:
                            retry_content.append({
                                "type": "input_text",
                                "text": f"[{frame.get('time_str', '')} 시점의 칠판/PPT 화면]",
                            })
                            retry_content.append({
                                "type": "input_image",
                                "image_url": _image_file_to_data_url(filepath),
                            })
                        except Exception as e:
                            print(f"[GPT 요약 재생성] 이미지 첨부 실패: {e}")

                retry_response = client.responses.create(
                    model=OPENAI_FINAL_MODEL_NAME,
                    input=[
                        {
                            "role": "user",
                            "content": retry_content,
                        }
                    ],
                    reasoning={
                        "effort": OPENAI_FINAL_REASONING_EFFORT,
                    },
                    max_output_tokens=OPENAI_FINAL_MAX_OUTPUT_TOKENS,
                )
                _log_openai_response_metadata(
                    retry_response,
                    label="GPT 최종 요약 재생성 응답",
                )
                retry_text = _extract_openai_output_text(retry_response)

                if retry_text:
                    print(f"[GPT 최종 요약 재생성 길이] {len(retry_text)}자")
                    return retry_text.strip()

                print("[GPT 경고] 재생성 응답 text가 비어 있어 최초 결과를 사용합니다.")
                return text.strip()

            print(f"[GPT 경고] 멀티모달 응답 text가 비어 있습니다. attempt={attempt}")
            return ""

        except Exception as e:
            last_error = e
            error_text = str(e)

            print(f"[GPT 멀티모달 호출 에러] attempt={attempt} / {error_text}")

            is_retryable = (
                "429" in error_text
                or "rate limit" in error_text.lower()
                or "500" in error_text
                or "502" in error_text
                or "503" in error_text
                or "504" in error_text
                or "timeout" in error_text.lower()
                or "temporarily" in error_text.lower()
            )

            if is_retryable:
                print(f"[GPT 재시도 대기] {wait_seconds}초 후 재시도합니다.")
                time.sleep(wait_seconds)
                continue

            raise e

    print(f"[GPT 멀티모달 최종 실패] 모든 재시도 실패: {last_error}")
    raise last_error


def generate_text_by_selected_api(prompt, summary_api="gemini"):
    """선택된 요약 API에 따라 Gemini 또는 GPT를 호출한다."""
    summary_api = normalize_summary_api(summary_api)

    if summary_api == "gpt":
        return openai_generate_text(prompt)

    return gemini_generate_text(prompt)


# =========================
# 이미지 마커 처리
# =========================

def replace_image_markers_with_html(ai_summary_text, lecture_id):
    """AI 요약문에 남겨진 [IMG: MM:SS] 마커를 실제 이미지 HTML로 치환한다.

    - **[IMG: 02:12]** 처럼 마크다운 bold로 감싼 경우도 처리한다.
    - 실제 파일이 없는 이미지 마커는 제거한다.
    - 이미지 HTML 앞뒤에 줄바꿈을 넣어 summary.html이 별도 블록으로 인식하게 한다.
    """
    text = ai_summary_text or ""

    pattern = r"\*{0,2}\[IMG:\s*(\d{1,3}:\d{2})\s*\]\*{0,2}"

    def match_to_img_tag(match):
        time_str = match.group(1).strip()
        minute_part = time_str.split(":")[0]

        if len(minute_part) == 1:
            safe_time_str = time_str.zfill(5).replace(":", "_")
        else:
            safe_time_str = time_str.replace(":", "_")

        filename = f"frame_{safe_time_str}.jpg"
        filepath = os.path.join(
            settings.MEDIA_ROOT,
            "lecture_frames",
            str(lecture_id),
            filename,
        )

        if not os.path.exists(filepath):
            print(f"[이미지 마커 치환] 파일 없음, 마커 제거: {filepath}")
            return ""

        image_url = f"/media/lecture_frames/{lecture_id}/{filename}"

        html = (
            f'\n\n<div class="timeline-image-wrapper" '
            f'style="text-align:center; margin: 25px 0; padding: 15px; '
            f'background-color:#f8f9fa; border-radius:12px;">'
            f'<img src="{image_url}" alt="{time_str} 강의화면" '
            f'style="max-width:90%; border-radius:8px; '
            f'box-shadow:0 4px 6px rgba(0,0,0,0.1);">'
            f'<p style="color:#6c757d; font-size:0.9em; '
            f'margin-top:10px; font-weight:bold;">▶ {time_str} 시점 판서 및 자료 참조</p>'
            f'</div>\n\n'
        )

        return html

    text = re.sub(pattern, match_to_img_tag, text)

    text = text.replace(
        '**<div class="timeline-image-wrapper"',
        '<div class="timeline-image-wrapper"',
    )
    text = text.replace("</div>**", "</div>")

    text = text.replace(
        '<div class="timeline-image-wrapper"',
        '\n\n<div class="timeline-image-wrapper"',
    )
    text = text.replace("</div>", "</div>\n\n")

    return text


# =========================
# 청크 요약 프롬프트
# =========================

def get_subject_chunk_prompt(subject_code, chunk_text):
    """과목별 특성에 맞게 청크 요약 프롬프트를 생성한다."""
    subject_code = normalize_subject_code(subject_code)

    base_instructions = """
[공통 요구사항]
- 모든 문장은 '~했다', '~이다', '~하다' 형식의 객관적인 평어체로 작성할 것.
- 입력 텍스트 맨 앞에 [분:초 ~ 분:초] 형태의 구간 헤더가 있다면, 해당 구간 정보를 출력 맨 앞에 그대로 포함할 것.
- 본문 중간에 임의의 타임라인을 추가하지 말 것.
"""

    if subject_code == "auto":
        return f"""너는 대학생이 원본 강의를 듣지 않아도 내용을 이해할 수 있도록 돕는 강의 대체형 요약 AI이다.
아래는 사용자가 과목 유형을 직접 지정하지 않은 강의의 일부 구간이다.
이 구간의 내용을 지나치게 압축하지 말고, 중요한 설명·예시·원인·결과·순서·용어를 최대한 보존해서 정리하라.

{base_instructions}

[AI 자동 판단 기준]
- 역사/인문학 성격이 강하면 사건의 원인, 전개, 결과, 인물, 제도, 조약, 시대적 의미를 중심으로 정리할 것.
- 프로그래밍/IT 성격이 강하면 개념 정의, 코드 흐름, 함수/클래스 역할, 실습 순서, 오류 가능 지점을 중심으로 정리할 것.
- 수학/과학 성격이 강하면 공식, 원리, 조건, 풀이 과정, 실험/현상, 단위와 예외를 중심으로 정리할 것.
- 일반 강의 성격이 강하면 핵심 주장, 근거, 사례, 결론을 중심으로 정리할 것.
- 과목이 혼합되어 있으면 가장 많이 다뤄진 흐름을 중심으로 정리하고, 보조 관점은 필요한 만큼 함께 기록할 것.

[구간 요약 목표]
- 이 구간 요약은 최종 요약의 재료가 되므로, 단순 압축보다 정보 보존을 우선한다.
- 강사가 설명한 핵심 개념, 세부 설명, 예시, 비교, 원인과 결과, 순서, 수치, 고유명사를 가능한 한 보존한다.
- 시험에 직접 나올 가능성이 낮아 보여도, 전체 흐름 이해에 필요한 내용이면 생략하지 않는다.
- 중복 표현은 줄이되, 서로 다른 정보는 삭제하지 않는다.
- 강의 내용에 없는 일반 지식이나 추측은 추가하지 않는다.
- 이 구간만으로 확실하지 않은 내용은 단정하지 말고 "이 구간만으로는 명확하지 않다"고 적는다.
- 분량은 고정하지 않는다. 내용이 많으면 길게 정리해도 된다.

[출력 방식]
- 이 구간의 핵심 흐름을 먼저 짧게 설명한다.
- 이어서 세부 내용을 불렛으로 정리한다.
- 불렛은 8개 이하로 억지로 줄이지 말고, 필요한 만큼 작성한다.
- 목록 기호는 '-'만 사용한다.
- '*' 목록 기호는 사용하지 않는다.
- '[AI가 분석한 강의 요약]' 같은 임시 문구는 절대 출력하지 않는다.

강의 내용:
{chunk_text}
"""

    if subject_code == "1":
        return f"""너는 역사와 인문학 강의를 정리하는 전문 강사 AI이다.
아래는 역사/인문학 강의의 일부 구간이다.
이 구간에 등장하는 인물, 사건, 시대적 배경, 핵심 개념, 인과관계를 중심으로 요약하라.

{base_instructions}

[과목별 요구사항]
- 인물, 사건, 제도, 사상, 시대 배경 등 구체적인 고유명사를 최대한 포함할 것.
- 사건의 원인, 전개 과정, 결과가 드러나도록 정리할 것.
- 개념 간의 인과관계와 흐름을 중심으로 설명할 것.

[출력 형식]
1. 구간 핵심 요약
2. 등장 고유명사 및 핵심 개념
3. 사건 또는 개념의 인과관계
4. 시험 포인트

강의 내용:
{chunk_text}
"""

    if subject_code == "2":
        return f"""너는 시니어 개발자이자 프로그래밍 강의를 정리하는 전문 강사 AI이다.
아래는 프로그래밍/IT 강의의 일부 구간이다.
개발자가 이해하고 실습에 적용할 수 있도록 기술 개념과 코드 흐름을 중심으로 요약하라.

{base_instructions}

[과목별 요구사항]
- 프로그래밍 언어, 프레임워크, 라이브러리, 함수, 클래스, 메서드 명칭을 명확히 정리할 것.
- 코드 실행 흐름, 데이터 흐름, 함수 호출 관계를 단계적으로 설명할 것.
- 오류 해결, 환경 설정, 구현 주의사항이 등장하면 반드시 포함할 것.
- 단순 줄글 요약보다 [개념] - [작동 방식] - [활용 방법] 중심으로 정리할 것.

[출력 형식]
1. 구간 핵심 요약
2. 핵심 기술 및 코드 흐름
3. 주요 함수/문법/개념
4. 실습 및 구현 주의사항

강의 내용:
{chunk_text}
"""

    if subject_code == "3":
        return f"""너는 수학 및 자연과학 강의를 정리하는 전문 강사 AI이다.
아래는 수학/과학 강의의 일부 구간이다.
개념, 공식, 원리, 문제 풀이 과정을 중심으로 요약하라.

{base_instructions}

[과목별 요구사항]
- 핵심 개념, 공식, 정리, 법칙이 등장하면 의미와 사용 조건을 명확히 정리할 것.
- 문제 풀이가 포함된 경우 문제 조건, 접근 방법, 풀이 과정, 결론을 구분할 것.
- 공식의 단순 암기보다 왜 그렇게 되는지 원리 중심으로 설명할 것.
- 수식은 텍스트로 읽기 쉽게 정리할 것.

[출력 형식]
1. 구간 핵심 개념
2. 공식 및 원리 설명
3. 문제 풀이 과정
4. 핵심 암기 포인트

강의 내용:
{chunk_text}
"""

    return f"""너는 다양한 분야의 강의를 정리하는 전문 강사 AI이다.
아래는 일반 강의의 일부 구간이다.
핵심 주제와 주요 개념, 전체 흐름을 중심으로 요약하라.

{base_instructions}

[과목별 요구사항]
- 강의의 핵심 주제와 세부 내용을 논리적으로 정리할 것.
- 등장하는 핵심 키워드와 개념을 명확히 설명할 것.
- 도입, 전개, 결론 흐름이 보이도록 구성할 것.

[출력 형식]
1. 구간 핵심 요약
2. 주요 개념 정리
3. 내용 전개 흐름
4. 복습 포인트

강의 내용:
{chunk_text}
"""


def summarize_chunk(chunk_text, subject_code="auto", summary_api="gemini"):
    """단일 청크를 과목별 프롬프트로 선택된 API를 사용해 요약한다."""
    try:
        prompt = get_subject_chunk_prompt(subject_code, chunk_text)
        result = generate_text_by_selected_api(prompt, summary_api=summary_api)

        if not result:
            return f"[부분 요약 생성 실패]\n{chunk_text}"

        return result

    except Exception as e:
        api_label = get_summary_model_label(summary_api)
        print(f"{api_label} chunk 요약 에러:", e)
        return f"[부분 요약 생성 실패]\n{chunk_text}"


# =========================
# 최종 요약 및 타임라인 추출
# =========================

def _dedup_summary(text):
    """Gemini가 섹션을 반복 출력한 경우 마지막 완성본만 남긴다."""
    text = (text or "").strip()
    marker = "1. 심층 배경 및 전체 요약"

    if text.count(marker) > 1:
        idx = text.rfind(marker)
        if idx > 0:
            text = text[idx:]

    return text.strip()


_SECTION_3_START = "3. 꼭 알아야 할"
_SECTION_4_START = "4. 핵심"
_TS = r"(\d{1,3}\s*:\s*\d{1,2}(?:\s*:\s*\d{1,2})?)"
_KEYWORD_TIMELINE_LINE_PATTERNS = (
    re.compile(rf"(?m)^\s*\*\s*\*\*(.+?)\s*\(\s*{_TS}\s*\)\s*\*\*\s*$"),
    re.compile(rf"(?m)^\s*-\s*\*\*(.+?)\s*\(\s*{_TS}\s*\)\s*\*\*\s*$"),
    re.compile(rf"(?m)^\s*-\s*(.+?)\s*\(\s*{_TS}\s*\)\s*$"),
    re.compile(rf"(?m)^\s*\*\s+(?!\*\*)(.+?)\s*\(\s*{_TS}\s*\)\s*$"),
    re.compile(rf"(?m)^\s*\*\*(.+?)\s*\(\s*{_TS}\s*\)\s*\*\*\s*$"),
)


def _normalize_summary_keyword(raw):
    """요약 타임라인에서 추출한 키워드 문자열을 정리한다."""
    s = (raw or "").strip()
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(":").strip()
    return s


def _extract_section_3(summary_text):
    """최종 요약문에서 '3. 꼭 알아야 할...' 섹션 본문만 추출한다."""
    text = (summary_text or "").strip()

    if not text:
        return ""

    if "[AI가 분석한 강의 요약]" in text:
        _, _, text = text.partition("[AI가 분석한 강의 요약]")
        text = text.strip().lstrip("\n").strip()

    start_idx = text.find(_SECTION_3_START)

    if start_idx < 0:
        return ""

    section = text[start_idx:]
    end_idx = section.find(_SECTION_4_START)

    if end_idx > 0:
        section = section[:end_idx]

    return section.strip()


def extract_summary_timeline(summary_text):
    """요약문 3번 섹션에서 '키워드 (분:초)' 형식의 타임라인만 추출한다."""
    section = _extract_section_3(summary_text)

    if not section:
        return ""

    candidates = []

    for pattern in _KEYWORD_TIMELINE_LINE_PATTERNS:
        for match in pattern.finditer(section):
            keyword = _normalize_summary_keyword(match.group(1))
            timestamp = re.sub(r"\s+", "", match.group(2).strip())

            if not keyword or len(keyword) > 120:
                continue

            candidates.append((match.start(), keyword, timestamp))

    candidates.sort(key=lambda item: item[0])

    timeline_lines = []
    seen = set()

    for _, keyword, timestamp in candidates:
        key = (keyword, timestamp)

        if key in seen:
            continue

        seen.add(key)
        timeline_lines.append(f"{keyword} ({timestamp})")

    return "\n".join(timeline_lines)


# =========================
# 핵심 타임라인 이미지 자동 삽입
# =========================

def timestamp_to_seconds(timestamp):
    """M:SS 또는 H:MM:SS 형식의 문자열을 초 단위로 변환한다."""
    raw = (timestamp or "").strip()
    raw = raw.strip("()[]")
    raw = re.sub(r"\s+", "", raw)

    if not raw or raw == "근거없음":
        return None

    parts = raw.split(":")

    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None

    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds

    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return hours * 3600 + minutes * 60 + seconds

    return None


def parse_summary_timeline_items(summary_text, max_items=5):
    """요약문 3번 섹션에서 핵심 개념과 타임라인을 구조화한다.

    반환 예시:
    [
        {"keyword": "난징조약", "time_str": "05:10", "seconds": 310}
    ]
    """
    section = _extract_section_3(summary_text)

    if not section:
        return []

    items = []
    seen = set()

    for pattern in _KEYWORD_TIMELINE_LINE_PATTERNS:
        for match in pattern.finditer(section):
            keyword = _normalize_summary_keyword(match.group(1))
            raw_timestamp = match.group(2)
            seconds = timestamp_to_seconds(raw_timestamp)

            if not keyword or seconds is None:
                continue

            if len(keyword) > 80:
                continue

            key = (keyword, seconds)

            if key in seen:
                continue

            seen.add(key)
            items.append({
                "keyword": keyword,
                "time_str": format_seconds_to_mmss(seconds),
                "seconds": seconds,
            })

    items.sort(key=lambda item: item["seconds"])

    if max_items and max_items > 0:
        return items[:max_items]

    return items


def _frame_seconds(frame):
    """timeline_frames 항목에서 초 단위 시간을 가져온다."""
    if not frame:
        return None

    if frame.get("seconds") is not None:
        try:
            return int(round(float(frame.get("seconds"))))
        except (TypeError, ValueError):
            return None

    return timestamp_to_seconds(frame.get("time_str", ""))


def select_best_frame_for_timeline(target_seconds, timeline_frames=None, max_distance_seconds=30):
    """핵심 개념 타임라인과 가장 가까운 프레임을 고른다."""
    if target_seconds is None or not timeline_frames:
        return None

    best_frame = None
    best_distance = None

    for frame in timeline_frames:
        filepath = frame.get("filepath")

        if not filepath or not os.path.exists(filepath):
            continue

        seconds = _frame_seconds(frame)

        if seconds is None:
            continue

        distance = abs(seconds - target_seconds)

        if distance <= max_distance_seconds:
            if best_frame is None or distance < best_distance:
                best_frame = frame
                best_distance = distance

    return best_frame


def _timeline_image_html(lecture_id, frame, keyword="", display_time=""):
    """선택된 프레임을 요약문에 삽입할 HTML로 변환한다."""
    if not lecture_id or not frame:
        return ""

    filepath = frame.get("filepath")

    if not filepath or not os.path.exists(filepath):
        return ""

    filename = os.path.basename(filepath)
    image_url = f"/media/lecture_frames/{lecture_id}/{filename}"
    time_str = display_time or frame.get("time_str") or ""
    caption_keyword = (keyword or frame.get("keyword") or "").strip()

    if caption_keyword:
        caption = f"▶ {time_str} 시점 핵심 화면: {caption_keyword}"
    else:
        caption = f"▶ {time_str} 시점 강의 화면"

    return (
        f'\n\n<div class="timeline-image-wrapper" '
        f'style="text-align:center; margin: 25px 0; padding: 15px; '
        f'background-color:#f8f9fa; border-radius:12px;">'
        f'<img src="{image_url}" alt="{caption_keyword or time_str} 강의화면" '
        f'style="max-width:90%; border-radius:8px; '
        f'box-shadow:0 4px 6px rgba(0,0,0,0.1);">'
        f'<p style="color:#6c757d; font-size:0.9em; '
        f'margin-top:10px; font-weight:bold;">{caption}</p>'
        f'</div>\n\n'
    )


def _line_has_timeline_seconds(line, target_seconds):
    """요약문 한 줄에 목표 초와 같은 타임라인이 있는지 확인한다."""
    for match in re.finditer(r"\(\s*(\d{1,3}\s*:\s*\d{1,2}(?:\s*:\s*\d{1,2})?)\s*\)", line):
        seconds = timestamp_to_seconds(match.group(1))

        if seconds == target_seconds:
            return True

    return False


def inject_core_timeline_images(summary_text, timeline_frames=None, lecture_id=None, max_images=4):
    """요약의 핵심 개념 타임라인에 맞춰 영상 프레임 이미지를 자동 삽입한다.

    기존 Gemini가 [IMG: MM:SS] 마커를 넣은 경우에는 이미 이미지 HTML이 생성되므로
    중복 삽입하지 않는다. 마커가 없거나 부족할 때 서버가 직접 핵심 개념 줄 아래에
    가장 가까운 프레임 이미지를 넣는다.
    """
    text = summary_text or ""

    if not text.strip() or not lecture_id or not timeline_frames:
        return text

    if text.count('class="timeline-image-wrapper"') >= max_images:
        return text

    items = parse_summary_timeline_items(text, max_items=max_images * 2)

    if not items:
        return text

    inserted_count = text.count('class="timeline-image-wrapper"')
    used_files = set()
    lines = text.splitlines()
    output_lines = []

    for line in lines:
        output_lines.append(line)

        if inserted_count >= max_images:
            continue

        stripped = line.strip()

        if not stripped:
            continue

        for item in items:
            keyword = item["keyword"]
            seconds = item["seconds"]
            time_str = item["time_str"]

            if keyword not in stripped:
                continue

            if not _line_has_timeline_seconds(stripped, seconds):
                continue

            best_frame = select_best_frame_for_timeline(
                target_seconds=seconds,
                timeline_frames=timeline_frames,
                max_distance_seconds=45,
            )

            if not best_frame:
                continue

            filepath = best_frame.get("filepath")

            if filepath in used_files:
                continue

            html = _timeline_image_html(
                lecture_id=lecture_id,
                frame=best_frame,
                keyword=keyword,
                display_time=time_str,
            )

            if not html:
                continue

            output_lines.append(html)
            used_files.add(filepath)
            inserted_count += 1
            break

    return "\n".join(output_lines)


def make_final_summary(
    chunk_summaries_text,
    timeline_frames=None,
    subject_code="auto",
    summary_api="gemini",
    source_char_count=None,
):
    """여러 부분 요약을 결합해 과목 맞춤형 최종 요약을 수행한다.

    source_char_count에는 최종 요약의 재료인 청크 요약 길이가 아니라
    STT 전체 원문의 글자 수를 전달한다. 이 값을 기준으로 요약 분량을 자동 조절한다.
    """
    try:
        subject_code = normalize_subject_code(subject_code)

        if not source_char_count:
            source_char_count = len(chunk_summaries_text or "")

        summary_plan = build_adaptive_summary_plan(source_char_count)
        target_min_chars = summary_plan["target_min_chars"]
        target_max_chars = summary_plan["target_max_chars"]
        overview_paragraph_max = summary_plan["overview_paragraph_max"]
        concept_max = summary_plan["concept_max"]

        print(
            f"[적응형 최종 요약] STT 원문={summary_plan['source_chars']:,}자, "
            f"청크 병합={len(chunk_summaries_text or ''):,}자, "
            f"권장 최종={target_min_chars:,}~{target_max_chars:,}자, "
            f"전체흐름 최대={overview_paragraph_max}문단, "
            f"필수개념 최대={concept_max}개"
        )

        adaptive_length_instruction = f"""
[적응형 요약 분량 규칙]
- STT 전체 원문 분량은 약 {summary_plan['source_chars']:,}자입니다.
- 최종 요약은 약 {target_min_chars:,}~{target_max_chars:,}자를 권장합니다.
- 위 범위를 크게 초과하지 마세요. 원문이 짧거나 정보가 적으면 분량을 채우기 위해 같은 말을 반복하거나 강의에 없는 배경지식을 추가하지 마세요.
- 반대로 원문에 실제로 존재하는 중요한 개념, 예시, 수치, 비교, 순서, 원인과 결과는 분량을 줄이기 위해 임의로 삭제하지 마세요.
- 1번 전체 흐름은 강의 정보량에 맞춰 최대 {overview_paragraph_max}개 문단까지만 작성하세요. 짧은 강의는 1개 문단으로 끝내도 됩니다.
- 3번 필수 개념은 실제로 중요한 항목만 최대 {concept_max}개까지 선정하세요. 개수를 채우기 위해 중요하지 않은 항목을 추가하지 마세요.
- 최종 결과는 반드시 1~4번 섹션을 모두 완성하고, 마지막 문장을 완결한 상태로 종료하세요.
"""

        image_time_list = []

        if timeline_frames:
            image_time_list = [
                frame.get("time_str", "")
                for frame in timeline_frames
                if frame.get("time_str")
            ]

        if image_time_list:
            image_time_text = "\n".join([f"- {time}" for time in image_time_list])
        else:
            image_time_text = "제공된 이미지 없음"

        multimodal_instruction = f"""
[제공된 시각 자료 시간 목록]
아래 시간의 이미지 파일만 실제로 제공됩니다.
이미지 마커를 사용할 경우 반드시 아래 시간 목록에 있는 시간만 사용하세요.

{image_time_text}

[시각 자료 삽입 규칙]
- 2번 섹션에서 제공된 이미지가 설명에 직접 도움이 될 때만 [IMG: 분:초] 마커를 삽입하세요.
- [IMG: 분:초] 마커는 반드시 [제공된 시각 자료 시간 목록]에 있는 시간만 사용하세요.
- 제공되지 않은 시간의 [IMG: 분:초] 마커는 절대 만들지 마세요.
- 마커를 굵게 표시하지 마세요. 즉, **[IMG: 02:12]** 처럼 쓰지 말고 [IMG: 02:12] 형식만 사용하세요.

[공통 출력 규칙]
- 세 번째 섹션의 제목은 반드시 "3. 꼭 알아야 할 필수 개념"으로 시작해야 합니다.
- 세 번째 섹션 안의 각 항목은 반드시 "- 키워드명 (분:초)" 형식이어야 합니다.
- 네 번째 섹션의 제목은 반드시 "4. 핵심"으로 시작해야 합니다.
- 같은 섹션 헤더를 두 번 이상 반복하지 말 것.
- 최종 완성본을 한 번만 출력할 것.
- HTML 태그를 직접 출력하지 말 것.
- 모든 문장의 끝맺음은 '~했다', '~이다', '~하다' 형식의 객관적인 평어체로 통일할 것.
- (분:초) 형식의 일반 타임라인은 오직 3번 섹션의 키워드 옆에만 표기할 것.

{adaptive_length_instruction}
"""

        if subject_code == "auto":
            subject_prompt = f"""당신은 대학생이 원본 강의를 듣지 않아도 강의 내용을 이해할 수 있도록 돕는 강의 대체형 요약 AI입니다.
다음은 사용자가 과목 유형을 직접 지정하지 않은 강의의 구간별 요약 내용입니다.
최종 목표는 짧은 요약문이 아니라, 강의를 듣지 않은 사용자도 전체 흐름과 핵심 내용을 손실 없이 이해할 수 있는 상세 강의 노트를 만드는 것입니다.

{multimodal_instruction}

[가장 중요한 목표]
- 사용자가 원본 강의를 듣지 않아도 강의의 종합적인 내용과 핵심 내용을 이해할 수 있어야 합니다.
- 단순히 짧게 줄이는 것보다 중요한 정보 보존을 우선합니다.
- 시험 대비에 필요한 내용뿐 아니라, 강의 흐름을 이해하는 데 필요한 배경·예시·원인·결과·비교·순서를 포함합니다.
- 원문의 정보량에 맞는 분량으로 작성합니다. 짧은 강의는 짧게, 정보가 많은 강의는 충분히 자세하게 정리합니다.

[AI 자동 판단 기준]
- 역사/인문학: 사건의 원인, 전개, 결과, 인물, 제도, 조약, 사상, 시대적 의미를 중심으로 정리합니다.
- 프로그래밍/IT: 개념 정의, 코드 흐름, 함수/클래스 역할, 데이터 흐름, 실습 순서, 오류 가능 지점을 중심으로 정리합니다.
- 수학/과학: 공식, 원리, 조건, 풀이 과정, 실험/현상, 단위, 예외를 중심으로 정리합니다.
- 일반 강의: 핵심 주장, 근거, 사례, 결론, 실제 적용 포인트를 중심으로 정리합니다.
- 혼합 강의라면 가장 중요한 흐름을 중심으로 잡고, 보조 성격도 필요한 만큼 반영합니다.

[정보 보존 규칙]
- 구간별 요약에 나온 중요한 사건, 개념, 인물, 용어, 순서, 수치, 예시, 비교, 원인과 결과를 최대한 유지합니다.
- 같은 의미의 반복은 합치되, 서로 다른 정보는 삭제하지 않습니다.
- 결론만 쓰지 말고, 왜 그런 결론이 나왔는지 과정도 설명합니다.
- 강사가 강조한 흐름이나 논리 전개가 보이도록 작성합니다.
- 강의에 없는 일반 지식, 배경지식, 추측성 설명을 억지로 추가하지 않습니다.
- 불확실한 내용은 단정하지 말고 "강의 내용만으로는 명확하지 않음"이라고 표시합니다.

[문체와 형식]
- Markdown 형식으로 작성합니다.
- 목록 기호는 '-'만 사용합니다.
- '*' 목록 기호는 사용하지 않습니다.
- '[AI가 분석한 강의 요약]' 같은 임시 문구는 절대 출력하지 않습니다.
- 문장은 너무 짧게 끊지 말고, 사용자가 강의 내용을 따라갈 수 있을 정도로 설명합니다.
- 단순 키워드 나열이 아니라, 개념과 흐름을 연결해서 설명합니다.

[출력 구조]
아래 4개 큰 섹션은 유지합니다.
단, 2번 섹션은 강의 내용을 충분히 복원하는 가장 중요한 부분이므로 자세히 작성합니다.

1. 강의 유형 판단 및 전체 흐름
- 판단한 강의 유형을 한 줄로 밝힙니다.
- 그렇게 판단한 근거를 강의 내용 기준으로 2~3개 제시합니다.
- 강의 전체가 어떤 문제의식에서 시작해 어떤 결론으로 이어지는지 정보량에 맞춰 설명합니다. 짧은 강의는 불필요하게 문단 수를 늘리지 않습니다.

2. 강의 대체 상세 정리
- 이 섹션은 가장 중요합니다.
- 사용자가 강의를 듣지 않아도 이해할 수 있도록 자세히 작성합니다.
- 고정된 [도입]-[핵심 개념]-[세부 설명]-[정리] 틀에 억지로 맞추지 말고, 강의 흐름에 맞게 소제목을 자유롭게 구성합니다.
- 역사/인문학이면 배경 → 원인 → 전개 → 결과 → 영향 → 이후 변화 흐름을 우선합니다.
- 프로그래밍/IT이면 개념 → 동작 방식 → 코드/구현 흐름 → 데이터 흐름 → 오류/주의사항 흐름을 우선합니다.
- 수학/과학이면 개념 → 공식/원리 → 적용 조건 → 풀이 과정 → 실수 포인트 흐름을 우선합니다.
- 일반 강의이면 핵심 주장 → 근거 → 사례 → 결론 → 적용 포인트 흐름을 우선합니다.
- 각 소제목 아래에는 단순 한두 문장이 아니라, 실제 강의 내용을 복원하듯 충분히 설명합니다.
- 제공된 이미지가 설명에 직접 도움이 되는 경우에만 [IMG: 분:초] 마커를 삽입합니다.

3. 꼭 알아야 할 필수 개념
- 시험 대비와 강의 이해에 실제로 중요한 키워드만 선정합니다. 위의 적응형 요약 분량 규칙에서 제시한 최대 개수를 넘기지 않습니다.
- 각 항목은 반드시 아래 형식을 지킵니다.
- 키워드명 (분:초)
  - 의미: 강의 내용 기준의 핵심 정의 또는 설명
  - 맥락: 이 개념이 강의 흐름에서 왜 중요한지 설명
  - 시험 포인트: 암기할 점, 헷갈리기 쉬운 점, 연결 개념 중 필요한 내용
- 시간 정보가 애매하면 구간별 요약에 나온 가장 가까운 시간을 사용합니다.
- 근거 있는 시간이 없으면 시간을 억지로 만들지 않습니다.

4. 핵심 출제 포인트
- 실제 시험 문제로 바뀔 수 있는 문장 위주로 정리합니다.
- 무엇을 묻기 쉬운지 구체적으로 작성합니다.
- 비교해서 외울 내용, 순서로 외울 내용, 원인과 결과로 외울 내용을 구분해서 작성합니다.
- 단답형, 서술형, 객관식으로 바뀔 수 있는 포인트를 골고루 포함합니다.
- 마지막에는 "한 줄 최종 정리"를 1문장으로 덧붙입니다.
"""

        elif subject_code == "1":
            subject_prompt = f"""당신은 역사와 인문학 강의를 정리하는 전문 강사 AI입니다.
다음은 유튜브 역사/인문학 강의의 구간별 분석 내용입니다.
이 내용을 바탕으로 인물, 사건, 시대 배경, 핵심 개념, 인과관계를 놓치지 않는 심층 분석 노트를 작성하세요.

{multimodal_instruction}

[과목별 요구사항]
- 인물, 연도, 지명, 사건, 사상, 제도, 조약 등 구체적인 고유명사를 최대한 포함할 것.
- 사건이나 개념의 원인, 전개 과정, 결과가 드러나도록 작성할 것.
- 단순 나열이 아니라 역사적 흐름과 인과관계 중심으로 설명할 것.

[출력 양식]
1. 심층 배경 및 전체 요약
- 강의가 다루는 시대적 배경과 전체 흐름을 정보량에 맞춰 정리할 것. 짧은 강의는 불필요하게 문단 수를 늘리지 말 것.

2. 흐름별 상세 전개
- [도입] - [전개] - [위기/절정] - [결말/영향] 단계로 나누어 설명할 것.
- 제공된 이미지가 설명에 직접 도움이 되는 경우에만 [IMG: 분:초] 마커를 삽입할 것.

3. 꼭 알아야 할 필수 개념 및 고유명사 사전
- 핵심 키워드는 실제로 중요한 항목만 선정하고, 위의 적응형 요약 분량 규칙에서 제시한 최대 개수를 넘기지 말 것.
- 키워드 이름 (분:초)
  - 구체적 의미 및 발생 원인
  - 역사적 결과 및 영향

4. 핵심 출제 포인트
- 인과관계, 시대별 변화, 개념 비교, 사건의 결과를 불렛포인트로 정리할 것.
"""

        elif subject_code == "2":
            subject_prompt = f"""당신은 시니어 개발자이자 프로그래밍/IT 강의를 정리하는 전문 강사 AI입니다.
다음은 프로그래밍/IT 강의의 구간별 분석 내용입니다.
이 내용을 바탕으로 개발자가 실습과 복습에 활용할 수 있는 기술 요약 노트를 작성하세요.

{multimodal_instruction}

[과목별 요구사항]
- 프로그래밍 언어, 프레임워크, 라이브러리, 함수, 클래스, 메서드, 명령어를 명확히 정리할 것.
- 코드 실행 흐름, 데이터 흐름, 함수 호출 관계, 아키텍처 구조를 단계적으로 설명할 것.
- 환경 설정, 오류 해결, 구현 주의사항, 실무 팁이 있으면 반드시 포함할 것.
- 전문 용어는 원래 영문 표기 또는 보편적인 한글 표기를 유지할 것.

[출력 양식]
1. 강의 개요 및 기술 스택
- 강의의 핵심 목표와 사용된 기술을 정보량에 맞춰 요약할 것. 짧은 강의는 1개 문단으로 끝내도 됨.

2. 핵심 기술 및 로직 전개
- [환경 설정/도입] - [핵심 문법 및 로직] - [실전 구현 방식] - [주의사항 및 트러블슈팅] 단계로 정리할 것.
- 제공된 이미지가 코드 흐름이나 화면 설명에 도움이 되는 경우에만 [IMG: 분:초] 마커를 삽입할 것.

3. 꼭 알아야 할 필수 개념 및 함수 사전
- 중요한 IT 개념, 함수명, 클래스명, 명령어만 정리하고, 위의 적응형 요약 분량 규칙에서 제시한 최대 개수를 넘기지 말 것.
- 키워드/함수명 (분:초)
  - 정의 및 작동 원리
  - 실무 적용 주의사항

4. 핵심 실무 적용 포인트
- 개발 시 주의사항, 에러 발생 포인트, 실무 팁을 불렛포인트로 정리할 것.
"""

        elif subject_code == "3":
            subject_prompt = f"""당신은 수학 및 자연과학 강의를 정리하는 전문 강사 AI입니다.
다음은 수학/과학 강의의 구간별 분석 내용입니다.
이 내용을 바탕으로 개념, 공식, 원리, 문제 풀이 과정이 드러나는 복습 노트를 작성하세요.

{multimodal_instruction}

[과목별 요구사항]
- 핵심 개념, 공식, 정리, 법칙이 등장하면 의미와 사용 조건을 명확히 작성할 것.
- 강사가 문제를 풀이했다면 문제 조건, 접근 방법, 풀이 과정, 결론을 구분하여 정리할 것.
- 공식의 단순 결과보다 도출 원리와 적용 방법을 설명할 것.
- 수식은 텍스트로 읽기 쉽게 표현할 것.

[출력 양식]
1. 핵심 개념 및 공식 정리
- 강의에서 다룬 주요 개념과 공식, 원리를 정리할 것.

2. 실전 문제 해설 및 개념 전개
- [문제 내용] - [접근 방법] - [풀이 과정] - [최종 결론] 흐름으로 정리할 것.
- 제공된 이미지가 문제 풀이 또는 공식 설명에 도움이 되는 경우에만 [IMG: 분:초] 마커를 삽입할 것.

3. 꼭 알아야 할 필수 개념 및 주요 공식 사전
- 실제로 중요한 공식, 법칙, 개념만 정리하고, 위의 적응형 요약 분량 규칙에서 제시한 최대 개수를 넘기지 말 것.
- 공식 및 개념 이름 (분:초)
  - 구체적 의미 및 유도 원리
  - 활용 방법 및 특징

4. 핵심 암기 공식 및 문제 풀이 포인트
- 자주 출제되는 공식, 실수하기 쉬운 부분, 풀이 전략을 불렛포인트로 정리할 것.
"""

        else:
            subject_prompt = f"""당신은 다양한 분야의 강의를 정리하는 전문 강사 AI입니다.
다음은 일반 강의의 구간별 분석 내용입니다.
이 내용을 바탕으로 핵심 주제와 주요 개념이 한눈에 들어오는 요약 노트를 작성하세요.

{multimodal_instruction}

[과목별 요구사항]
- 강의의 핵심 주제와 전체 흐름을 논리적으로 정리할 것.
- 등장하는 핵심 키워드와 개념을 명확히 설명할 것.
- 단순 요약보다 도입, 전개, 결론 흐름이 보이도록 구성할 것.

[출력 양식]
1. 심층 배경 및 전체 요약
- 강의의 핵심 주제와 전체 흐름을 정보량에 맞춰 정리할 것. 짧은 강의는 불필요하게 문단 수를 늘리지 말 것.

2. 흐름별 상세 전개
- [도입] - [주요 개념 전개] - [핵심 결론] 순으로 정리할 것.
- 제공된 이미지가 설명에 직접 도움이 되는 경우에만 [IMG: 분:초] 마커를 삽입할 것.

3. 꼭 알아야 할 필수 개념 및 고유명사 사전
- 실제로 중요한 키워드만 정리하고, 위의 적응형 요약 분량 규칙에서 제시한 최대 개수를 넘기지 말 것.
- 키워드 이름 (분:초)
  - 구체적 의미 및 특징
  - 주요 영향 및 결론

4. 핵심 요약 포인트
- 강의 전체에서 강조된 핵심 포인트를 불렛포인트로 정리할 것.
"""

        prompt_text = subject_prompt + f"\n\n부분 요약 리스트:\n{chunk_summaries_text}"
        summary_api = normalize_summary_api(summary_api)

        if summary_api == "gpt":
            raw = openai_generate_multimodal_text(
                prompt_text=prompt_text,
                timeline_frames=timeline_frames,
                target_max_chars=target_max_chars,
            )
        else:
            contents_to_send = [prompt_text]

            if timeline_frames:
                for frame in timeline_frames:
                    try:
                        img = PIL.Image.open(frame["filepath"])
                        contents_to_send.append(f"[{frame['time_str']} 시점의 칠판/PPT 화면]")
                        contents_to_send.append(img)
                    except Exception as e:
                        print(f"[Gemini 요약 엔지니어링] 이미지 임베딩 실패: {e}")

            raw = gemini_generate_text(contents_to_send)

        if not raw:
            return chunk_summaries_text

        return _dedup_summary(raw)

    except Exception as e:
        print(f"{get_summary_model_label(summary_api)} 최종 요약 에러:", e)
        return chunk_summaries_text


# =========================
# 요약문 개념 사전 추출
# =========================

def extract_concept_entries_from_summary(summary_text):
    """최종 요약의 '3. 꼭 알아야 할 필수 개념 및 고유명사 사전'에서 개념 정보를 추출한다.

    추출 결과:
    [
        {
            "keyword": "난징조약",
            "timeline": "(5:10)",
            "description": "구체적 의미 ... 역사적 결과 ..."
        }
    ]
    """
    text = clean_summary_text(summary_text)

    if not text:
        return []

    section_match = re.search(
        r"3\.\s*꼭\s*알아야\s*할\s*필수\s*개념.*?(?=\n\s*4\.\s*핵심|\Z)",
        text,
        re.DOTALL,
    )

    if section_match:
        concept_section = section_match.group(0)
    else:
        concept_section = text

    lines = concept_section.splitlines()

    entries = []
    current = None

    concept_line_pattern = re.compile(
        r"^\s*[-•]\s*(.+?)(\(\s*\d{1,3}\s*:\s*\d{1,2}(?:\s*:\s*\d{1,2})?\s*\))?\s*$"
    )

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        match = concept_line_pattern.match(line)

        if match:
            keyword = match.group(1).strip()
            timeline = (match.group(2) or "").strip()

            if (
                keyword.startswith("구체적")
                or keyword.startswith("역사적")
                or keyword.startswith("의미")
                or keyword.startswith("결과")
            ):
                if current:
                    current["description_parts"].append(keyword)
                continue

            if len(keyword) > 40:
                if current:
                    current["description_parts"].append(keyword)
                continue

            current = {
                "keyword": keyword,
                "timeline": timeline,
                "description_parts": [],
            }
            entries.append(current)
            continue

        if current:
            cleaned = re.sub(r"^\s*[-•]\s*", "", line).strip()
            if cleaned:
                current["description_parts"].append(cleaned)

    final_entries = []

    for entry in entries:
        keyword = (entry.get("keyword") or "").strip()
        timeline = (entry.get("timeline") or "").strip()
        description = " ".join(entry.get("description_parts") or []).strip()

        if not keyword:
            continue

        if len(description) < 15:
            continue

        final_entries.append(
            {
                "keyword": keyword,
                "timeline": timeline if timeline else "근거 없음",
                "description": description,
            }
        )

    return final_entries