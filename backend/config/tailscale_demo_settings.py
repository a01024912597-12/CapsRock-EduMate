from .settings import *

STATIC_ROOT = BASE_DIR / "staticfiles"

MIDDLEWARE = list(MIDDLEWARE)

whitenoise_middleware = "whitenoise.middleware.WhiteNoiseMiddleware"

if whitenoise_middleware not in MIDDLEWARE:
    security_index = MIDDLEWARE.index(
        "django.middleware.security.SecurityMiddleware"
    )
    MIDDLEWARE.insert(security_index + 1, whitenoise_middleware)

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}

# EDUMATE_CUSTOM_HOST
ALLOWED_HOSTS = list(ALLOWED_HOSTS)
if "edumate.taild143e6.ts.net" not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append("edumate.taild143e6.ts.net")

CSRF_TRUSTED_ORIGINS = list(CSRF_TRUSTED_ORIGINS)
if "https://edumate.taild143e6.ts.net" not in CSRF_TRUSTED_ORIGINS:
    CSRF_TRUSTED_ORIGINS.append(
        "https://edumate.taild143e6.ts.net"
    )
