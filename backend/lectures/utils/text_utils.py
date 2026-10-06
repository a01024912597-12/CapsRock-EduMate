import re
from kiwipiepy import Kiwi


kiwi = Kiwi()


def preprocess_text(text):
    """Whisper 전사 결과를 문장 단위로 정제한다."""
    filler_words = ["음", "어", "그니까", "약간", "이제", "뭐랄까"]

    text = re.sub(r"\s+", " ", text).strip()

    raw_sentences = kiwi.split_into_sents(text)

    processed_sentences = []
    seen_sentences = set()

    for sent in raw_sentences:
        tokens = kiwi.tokenize(sent.text.strip())

        refined_sent = ""

        for token in tokens:
            if token.form not in filler_words:
                if token.tag.startswith("J") or token.tag.startswith("E") or token.tag.startswith("X"):
                    refined_sent += token.form
                else:
                    refined_sent += " " + token.form

        refined_sent = refined_sent.strip()
        refined_sent = re.sub(r"\s+", " ", refined_sent)

        if len(refined_sent) > 10 and refined_sent not in seen_sentences:
            processed_sentences.append(refined_sent)
            seen_sentences.add(refined_sent)

    return processed_sentences

def chunk_text(sentences, chunk_size=10):
    """문장 리스트를 지정 개수 단위의 청크로 묶는다."""
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks

def clean_summary_text(summary_text):
    """요약문 앞의 안내 문구를 제거하고 실제 요약 본문만 정리한다."""
    text = (summary_text or "").strip()

    text = text.replace("[AI가 분석한 강의 요약]", "").strip()

    return text

def normalize_text_for_keyword(text):
    """키워드 매칭용 텍스트를 정리한다."""
    text = (text or "").strip()
    text = re.sub(r"\s+", "", text)
    return text

def strip_html_tags(text):
    """요약문 안의 이미지 HTML 등을 객관식 생성 전에 제거한다."""
    text = text or ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
