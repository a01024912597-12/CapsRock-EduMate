EduMate 2026-11-10 - GPT 기본 요약 패치

변경 파일 6개
1) backend/lectures/utils/option_utils.py
2) backend/lectures/views.py
3) backend/lectures/services/analysis_service.py
4) backend/lectures/services/summary_service.py
5) backend/templates/upload.html
6) backend/templates/includes/summary_request_form.html

변경 사항
- 요약 API의 기본값과 fallback을 Gemini -> GPT로 변경
- 강의 업로드 화면의 기본 선택을 GPT로 변경
- 강의 요약 재요청 화면의 기본 선택을 GPT로 변경
- Gemini 경로는 그대로 유지되어 사용자가 직접 선택하면 계속 사용 가능
- 기존 번역 문자열은 변경하지 않음
