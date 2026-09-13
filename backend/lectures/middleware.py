from django.conf import settings
from django.utils import translation


class PreferredLanguageMiddleware:
    """로그인 사용자는 UserProfile.preferred_language를 우선 적용한다."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, "user", None)
        if user is not None and getattr(user, "is_authenticated", False):
            lang = ""
            try:
                lang = (user.profile.preferred_language or "").strip()
            except Exception:
                lang = ""
            if lang in dict(settings.LANGUAGES):
                translation.activate(lang)
                request.LANGUAGE_CODE = lang
        return self.get_response(request)
