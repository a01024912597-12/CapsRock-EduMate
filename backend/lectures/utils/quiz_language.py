import re
from collections import Counter


LANG_KO = "ko"
LANG_EN = "en"
LANG_ZH = "zh-hans"


def detect_quiz_language(text):
    """요약문 자체를 기준으로 퀴즈 생성 언어를 판별한다."""
    text = text or ""
    lowered = text.lower()

    if any(marker in lowered for marker in (
        "lecture type and overall flow",
        "detailed lecture notes",
        "essential concepts",
        "key exam points",
    )):
        return LANG_EN

    if any(marker in text for marker in (
        "课程类型", "整体流程", "详细讲义", "核心概念", "考试重点", "关键考点",
    )):
        return LANG_ZH

    hangul_count = len(re.findall(r"[가-힣]", text))
    han_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))

    if han_count >= 20 and han_count > hangul_count * 2:
        return LANG_ZH
    if latin_count >= 80 and latin_count > hangul_count * 3:
        return LANG_EN
    return LANG_KO


def split_sentences_multilingual(text, language=None):
    """한국어 외 요약도 안정적으로 문장 단위로 분리한다."""
    text = re.sub(r"\s+", " ", text or "").strip()
    language = language or detect_quiz_language(text)
    if not text:
        return []

    if language == LANG_ZH:
        parts = re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+", text)
    else:
        parts = re.split(r"(?<=[.!?])\s+", text)

    return [part.strip() for part in parts if part and part.strip()]


def _dedupe_keep_order(values):
    seen = set()
    result = []
    for value in values:
        value = re.sub(r"\s+", " ", str(value or "")).strip(" -•:：,.;()[]{}")
        if len(value) < 2:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _clean_multilingual_quiz_source(text, language=None):
    """퀴즈 후보 추출 전에 HTML/CSS/마크다운 메타 문구를 제거한다."""
    text = text or ""
    language = language or detect_quiz_language(text)

    # 이미지 wrapper/style/url이 키워드 후보로 섞이지 않도록 먼저 제거한다.
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]*\]\((?:https?://|/)[^)]+\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:font-size|timeline-image-wrapper|font-family|line-height|border-radius|background-color)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:px|rem|rgba?|var|display|margin|padding|width|height|color|style|class)\b", " ", text, flags=re.IGNORECASE)

    # 캡처 이미지 설명은 문제 개념 후보에서 제외한다.
    text = re.sub(r"▶\s*\d{1,3}:\d{2}[^\n]*", " ", text)
    text = re.sub(r"\[IMG:\s*\d{1,3}:\d{2}\]", " ", text, flags=re.IGNORECASE)
    return text


def _is_useful_non_korean_keyword(value, language):
    value = re.sub(r"\s+", " ", str(value or "")).strip(" -•:：,.;()[]{}*_`")
    if not 2 <= len(value) <= 60:
        return False

    lowered = value.casefold()
    common_junk = {
        "ai lecture summary", "lecture summary", "lecture type", "overall flow",
        "detailed lecture notes", "essential concepts", "key exam points",
        "type", "basis", "meaning", "context", "exam point", "exam points",
        "short answer", "comparison", "sequence", "essay", "one-line final summary",
        "one line final summary", "summary", "notes", "result", "results",
        "课程摘要", "课程类型", "整体流程", "详细讲义", "核心概念", "考试重点",
        "关键考点", "含义", "语境", "考点", "一句话总结",
    }
    if lowered in common_junk:
        return False

    if any(token in lowered for token in (
        "font-size", "timeline-image", "wrapper", "background", "margin", "padding",
        "border", "display", "color:", "http://", "https://", ".jpg", ".png",
    )):
        return False

    # CSS/HTML 식별자처럼 보이는 긴 하이픈 연결 문자열은 제외한다.
    if re.fullmatch(r"[a-z]+(?:-[a-z]+){2,}", value):
        return False

    if language == LANG_EN:
        stopwords = {
            "the", "and", "for", "that", "with", "from", "this", "into", "than", "then",
            "when", "where", "which", "while", "have", "has", "had", "are", "was", "were",
            "will", "would", "should", "could", "can", "may", "might", "must", "not", "but",
            "about", "after", "before", "through", "using", "used", "user", "users", "lecture",
            "summary", "notes", "type", "basis", "meaning", "context", "exam", "point", "points",
            "result", "results", "work", "task", "tasks", "data", "file", "files", "one", "two",
            "its", "their", "they", "them", "you", "your", "our", "out", "only", "more", "also",
            "recurring", "finished", "verified", "repeated", "important", "central", "effective",
            "practical", "link", "code", "one-line", "ai",
        }
        if lowered in stopwords:
            return False
        if not re.search(r"[A-Za-z]", value):
            return False
        first_word = re.split(r"\s+", lowered, maxsplit=1)[0]
        if first_word in {"practical", "effective", "dedicated", "separate", "better", "reusable", "link", "delegates"}:
            return False
    else:
        zh_stop = {
            "课程", "内容", "讲义", "总结", "概念", "重点", "说明", "含义", "意义",
            "背景", "语境", "考点", "结果", "用户", "使用", "可以", "需要", "进行",
            "一个", "一种", "通过", "以及", "因此", "这是", "其中", "核心", "整体",
        }
        if value in zh_stop:
            return False

    return True


def extract_non_korean_keywords(text, language, concept_keywords=None):
    """영어/중국어 요약에서 실제 강의 개념 위주로 퀴즈 후보 용어를 추출한다."""
    text = _clean_multilingual_quiz_source(text, language)
    candidates = []

    # 1순위: 요약의 필수 개념 섹션에서 추출된 키워드.
    # 영어에서 여러 개념을 묶은 제목(Projects and Artifacts 등)은
    # 하나의 빈칸 정답으로 쓰지 않고 아래에서 개별 개념으로 분리한다.
    for value in concept_keywords or []:
        if language == LANG_EN and re.search(r"\s+(?:and|vs\.?|versus)\s+|[,;/|]", value, flags=re.IGNORECASE):
            continue
        if _is_useful_non_korean_keyword(value, language):
            candidates.append(value)

    # 2순위: GPT가 굵게 표시한 개념명. Type/Basis/Meaning 같은 구조 라벨은 제외한다.
    for match in re.finditer(r"\*\*([^*\n]{2,60})\*\*", text):
        raw = match.group(1).strip()
        if ":" in raw or "：" in raw:
            parts = re.split(r"[:：]", raw, maxsplit=1)
            # **Projects:**처럼 라벨 자체가 개념이면 라벨을 살리고,
            # **Label: value** 형태면 실제 value를 우선한다.
            term = (parts[1].strip() if len(parts) > 1 and parts[1].strip() else parts[0].strip())
        else:
            term = raw
        if _is_useful_non_korean_keyword(term, language):
            candidates.append(term)

    if language == LANG_EN:
        # 'Projects and Artifacts', 'Skills, Connectors, and MCP'처럼
        # 필수 개념 제목에 묶여 있는 실제 용어를 개별 후보로도 보존한다.
        for concept in concept_keywords or []:
            for part in re.split(r"\s+(?:and|vs\.?|versus)\s+|[,;/|]", concept, flags=re.IGNORECASE):
                part = part.strip(" -•:：,.;()[]{}*_`")
                if _is_useful_non_korean_keyword(part, language):
                    candidates.append(part)

        # 일반 문장의 첫 단어(Effective, Dedicated 등)가 오답 후보가 되지 않도록
        # 약어·다단어 고유명사·반복 등장 고유명사만 보충한다.
        acronyms = re.findall(r"\b[A-Z]{2,8}\b", text)
        multi_names = re.findall(r"\b[A-Z][A-Za-z0-9.+#'/-]{1,}(?:\s+[A-Z][A-Za-z0-9.+#'/-]{1,}){1,3}\b", text)
        candidates.extend(term for term in acronyms if _is_useful_non_korean_keyword(term, language))
        candidates.extend(term for term in multi_names if _is_useful_non_korean_keyword(term, language))

        single_caps = re.findall(r"\b[A-Z][A-Za-z0-9.+#'/-]{2,}\b", text)
        cap_counts = Counter(single_caps)
        for term, count in cap_counts.most_common():
            if count >= 2 and _is_useful_non_korean_keyword(term, language):
                candidates.append(term)

        # 후보가 매우 부족할 때만 반복 빈도가 높은 일반 명사를 제한적으로 보충한다.
        if len(_dedupe_keep_order(candidates)) < 6:
            stopwords = {
                "the", "and", "for", "that", "with", "from", "this", "into", "than", "then",
                "when", "where", "which", "while", "have", "has", "had", "are", "was", "were",
                "will", "would", "should", "could", "can", "may", "might", "must", "not", "but",
                "about", "after", "before", "through", "using", "used", "user", "users", "lecture",
                "summary", "notes", "type", "basis", "meaning", "context", "exam", "point", "points",
                "result", "results", "work", "task", "tasks", "data", "file", "files", "one", "two",
                "recurring", "finished", "verified", "effective", "central", "important",
                "practical", "link", "code", "one-line", "ai",
            }
            words = re.findall(r"\b[A-Za-z][A-Za-z']{3,}\b", text)
            counts = Counter(word for word in words if word.casefold() not in stopwords)
            for word, count in counts.most_common(40):
                if count >= 2 and _is_useful_non_korean_keyword(word, language):
                    candidates.append(word)

    elif language == LANG_ZH:
        latin_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9.+#'/-]{1,}(?:\s+[A-Za-z][A-Za-z0-9.+#'/-]{1,}){0,2}\b", text)
        candidates.extend(
            term for term in latin_terms
            if _is_useful_non_korean_keyword(term, language)
        )

        # 목록의 콜론 앞부분은 중국어 개념명인 경우가 많다.
        for line in text.splitlines():
            line = re.sub(r"^[\s#>*\-•]+", "", line).strip()
            if not line:
                continue
            head = re.split(r"[:：]", line, maxsplit=1)[0].strip()
            head = re.sub(r"[*_`]+", "", head).strip()
            if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9 /+.-]{2,28}", head) and _is_useful_non_korean_keyword(head, language):
                candidates.append(head)

        if len(_dedupe_keep_order(candidates)) < 6:
            runs = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
            counts = Counter(run for run in runs if _is_useful_non_korean_keyword(run, language))
            candidates.extend([run for run, count in counts.most_common(60) if count >= 2])

    return _dedupe_keep_order(candidates)


def localized_quiz_text(language, key, **kwargs):
    table = {
        LANG_KO: {
            "blank_prompt": "다음 빈칸에 알맞은 단어를 고르세요.\n{sentence}",
            "explanation": "본 문항은 강의 요약의 핵심 키워드 '{answer}'를 확인하기 위한 문제입니다.",
            "tf_true": "강의 본문 내용과 정확히 일치하는 명제입니다.",
            "tf_false": "강의 본문의 '{old}'(이)가 '{new}'(으)로 잘못 변경된 명제입니다.",
            "tf_false_generic": "강의 내용과 일치하지 않는 명제입니다.",
        },
        LANG_EN: {
            "blank_prompt": "Choose the word or term that best fills in the blank.\n{sentence}",
            "explanation": "This question checks the key lecture term '{answer}'.",
            "tf_true": "This statement matches the lecture summary.",
            "tf_false": "The lecture term '{old}' was incorrectly replaced with '{new}'.",
            "tf_false_generic": "This statement does not match the lecture summary.",
        },
        LANG_ZH: {
            "blank_prompt": "请选择最适合填入空格的词语或术语。\n{sentence}",
            "explanation": "本题用于确认课程摘要中的核心术语“{answer}”。",
            "tf_true": "该陈述与课程摘要内容一致。",
            "tf_false": "课程中的术语“{old}”被错误替换为“{new}”。",
            "tf_false_generic": "该陈述与课程摘要内容不一致。",
        },
    }
    lang_table = table.get(language, table[LANG_KO])
    return lang_table.get(key, "").format(**kwargs)
