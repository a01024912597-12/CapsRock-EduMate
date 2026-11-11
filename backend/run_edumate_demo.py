import os
import re
import mimetypes
from pathlib import Path
from wsgiref.util import FileWrapper

os.environ["DJANGO_SETTINGS_MODULE"] = "config.tailscale_demo_settings"

from django.conf import settings
from django.core.wsgi import get_wsgi_application
from waitress import serve

django_app = get_wsgi_application()
frame_root = (Path(settings.MEDIA_ROOT) / "lecture_frames").resolve()

frame_pattern = re.compile(
    r"^/media/lecture_frames/(\d+)/(verified_[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|webp))$",
    re.IGNORECASE,
)

def application(environ, start_response):
    path = environ.get("PATH_INFO", "")
    match = frame_pattern.fullmatch(path)

    if not match:
        return django_app(environ, start_response)

    if environ.get("REQUEST_METHOD") not in ("GET", "HEAD"):
        start_response("405 Method Not Allowed", [
            ("Content-Length", "0"),
        ])
        return [b""]

    image_path = (frame_root / match.group(1) / match.group(2)).resolve()

    if frame_root not in image_path.parents or not image_path.is_file():
        start_response("404 Not Found", [
            ("Content-Length", "0"),
        ])
        return [b""]

    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"

    start_response("200 OK", [
        ("Content-Type", mime),
        ("Content-Length", str(image_path.stat().st_size)),
        ("X-Content-Type-Options", "nosniff"),
        ("Cache-Control", "private, max-age=300"),
    ])

    if environ.get("REQUEST_METHOD") == "HEAD":
        return [b""]

    return FileWrapper(image_path.open("rb"), 65536)

if __name__ == "__main__":
    print("EduMate: http://127.0.0.1:8000")
    serve(
        application,
        host="127.0.0.1",
        port=8000,
        threads=4,
    )
