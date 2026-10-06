"""요약문 PDF보내기 (reportlab + 프로젝트 내장 Noto Sans KR)."""

import os
import re
import xml.sax.saxutils
from io import BytesIO
from pathlib import Path

from django.conf import settings
from django.utils import timezone
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

TIMELINE_IMAGE_PREFIX = '<div class="timeline-image-wrapper"'
PDF_FONT_NAME = "NotoSansKR"
BUNDLED_FONT_FILENAME = "NotoSansKR-Regular.ttf"


def get_bundled_korean_font_path():
    """프로젝트에 포함된 Noto Sans KR 경로를 반환한다."""
    return Path(settings.BASE_DIR) / "static" / "fonts" / BUNDLED_FONT_FILENAME


def register_pdf_font():
    """reportlab용 한글 폰트를 등록한다."""
    if PDF_FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return PDF_FONT_NAME

    font_path = get_bundled_korean_font_path()

    if not font_path.is_file():
        raise FileNotFoundError(
            f"PDF 한글 폰트가 없습니다: {font_path}. "
            f"static/fonts/{BUNDLED_FONT_FILENAME} 파일을 확인해 주세요."
        )

    pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, str(font_path)))
    pdfmetrics.registerFontFamily(
        PDF_FONT_NAME,
        normal=PDF_FONT_NAME,
        bold=PDF_FONT_NAME,
        italic=PDF_FONT_NAME,
        boldItalic=PDF_FONT_NAME,
    )
    return PDF_FONT_NAME


def pdf_escape(text):
    """Paragraph용 텍스트 이스케이프."""
    return xml.sax.saxutils.escape(text or "")


def prepare_summary_text(summary_text):
    """PDF용 요약 본문에서 안내 문구를 제거한다."""
    text = (summary_text or "").strip()
    return text.replace("[AI가 분석한 강의 요약]", "").strip()


def is_allowed_timeline_image_html(block):
    return (
        TIMELINE_IMAGE_PREFIX in block
        and "<img " in block
        and "/media/lecture_frames/" in block
    )


def image_path_from_timeline_html(block):
    """타임라인 이미지 HTML에서 로컬 파일 경로를 추출한다."""
    match = re.search(r'src="(/media/lecture_frames/[^"]+)"', block)

    if not match:
        return None

    relative = match.group(1)[len("/media/") :]
    path = os.path.join(settings.MEDIA_ROOT, relative.replace("/", os.sep))

    if os.path.isfile(path):
        return path

    return None


def image_caption_from_timeline_html(block):
    match = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)

    if not match:
        return ""

    caption = re.sub(r"<[^>]+>", "", match.group(1))
    return caption.strip()


def build_pdf_styles(font_name):
    """PDF 단락 스타일을 만든다."""
    return {
        "title": ParagraphStyle(
            "SummaryTitle",
            fontName=font_name,
            fontSize=18,
            leading=24,
            textColor="#111827",
            spaceAfter=8,
        ),
        "meta": ParagraphStyle(
            "SummaryMeta",
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            textColor="#6b7280",
            spaceAfter=16,
        ),
        "section": ParagraphStyle(
            "SummarySection",
            fontName=font_name,
            fontSize=13,
            leading=18,
            textColor="#1d4ed8",
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "SummaryBody",
            fontName=font_name,
            fontSize=11,
            leading=17,
            textColor="#1f2937",
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "SummaryBullet",
            fontName=font_name,
            fontSize=11,
            leading=17,
            textColor="#1f2937",
            leftIndent=14,
            bulletIndent=0,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "SummaryCaption",
            fontName=font_name,
            fontSize=9,
            leading=12,
            textColor="#6b7280",
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "footer": ParagraphStyle(
            "SummaryFooter",
            fontName=font_name,
            fontSize=9,
            leading=12,
            textColor="#9ca3af",
            spaceBefore=18,
        ),
    }


def build_summary_pdf_flowables(lecture):
    """요약 PDF 본문 flowable 목록을 만든다."""
    font_name = register_pdf_font()
    styles = build_pdf_styles(font_name)
    flowables = []

    flowables.append(Paragraph(pdf_escape(lecture.title or "강의 요약"), styles["title"]))

    if lecture.source_type == "file":
        source_line = (
            f"업로드 파일: {os.path.basename(lecture.video_file.name)}"
            if lecture.video_file
            else "업로드 파일: -"
        )
    else:
        source_line = f"유튜브: {lecture.youtube_url or '-'}"

    meta_lines = [source_line]

    if lecture.analyzed_at:
        analyzed_at = timezone.localtime(lecture.analyzed_at).strftime("%Y-%m-%d %H:%M")
        meta_lines.append(f"분석 완료: {analyzed_at}")

    flowables.append(Paragraph(pdf_escape("<br/>".join(meta_lines)), styles["meta"]))

    lines = prepare_summary_text(lecture.summary_text).split("\n")
    in_image_block = False
    image_buffer = []

    for raw in lines:
        trimmed = raw.strip()

        if trimmed.startswith(TIMELINE_IMAGE_PREFIX):
            in_image_block = True
            image_buffer = [raw]

            if "</div>" in trimmed:
                block = "\n".join(image_buffer)
                flowables.extend(_timeline_image_flowables(block, styles))
                image_buffer = []
                in_image_block = False

            continue

        if in_image_block:
            image_buffer.append(raw)

            if "</div>" in trimmed:
                block = "\n".join(image_buffer)
                flowables.extend(_timeline_image_flowables(block, styles))
                image_buffer = []
                in_image_block = False

            continue

        if not trimmed:
            flowables.append(Spacer(1, 4 * mm))
            continue

        section_match = re.match(r"^(\d+)\.\s+(.+)$", trimmed)
        if section_match:
            heading = f"{section_match.group(1)}. {section_match.group(2)}"
            flowables.append(Paragraph(pdf_escape(heading), styles["section"]))
            continue

        bullet_match = re.match(r"^[-•]\s+(.+)$", trimmed)
        if bullet_match:
            flowables.append(
                Paragraph(
                    f"• {pdf_escape(bullet_match.group(1))}",
                    styles["bullet"],
                )
            )
            continue

        flowables.append(Paragraph(pdf_escape(trimmed), styles["body"]))

    if in_image_block and image_buffer:
        flowables.extend(_timeline_image_flowables("\n".join(image_buffer), styles))

    flowables.append(Paragraph(pdf_escape("EduMate AI 강의 요약"), styles["footer"]))
    return flowables


def _timeline_image_flowables(block, styles):
    """타임라인 이미지 블록을 PDF flowable로 변환한다."""
    if not is_allowed_timeline_image_html(block):
        return []

    image_path = image_path_from_timeline_html(block)

    if not image_path:
        return []

    flowables = []

    try:
        image = Image(image_path)
        max_width = 160 * mm
        ratio = image.imageWidth / float(image.imageHeight or 1)
        image.drawWidth = min(max_width, image.imageWidth)
        image.drawHeight = image.drawWidth / ratio
        image.hAlign = "CENTER"
        flowables.append(Spacer(1, 3 * mm))
        flowables.append(image)
    except Exception as exc:
        print(f"[PDF 이미지 삽입 실패] {image_path}: {exc}")
        return flowables

    caption = image_caption_from_timeline_html(block)

    if caption:
        flowables.append(Paragraph(pdf_escape(caption), styles["caption"]))

    return flowables


def generate_summary_pdf_bytes(lecture):
    """강의 요약 PDF 바이트를 생성한다."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=lecture.title or "강의 요약",
    )
    doc.build(build_summary_pdf_flowables(lecture))
    return buffer.getvalue()


def build_summary_pdf_filename(lecture):
    """다운로드 파일명을 만든다."""
    raw_title = (lecture.title or "lecture-summary").strip()
    safe = re.sub(r'[\\/:*?"<>|]+', "_", raw_title)
    safe = re.sub(r"\s+", "_", safe).strip("._") or "lecture-summary"
    return f"{safe[:80]}-{lecture.id}.pdf"
