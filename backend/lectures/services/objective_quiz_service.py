import os
import re
import json
import random
import threading
from collections import Counter

import torch
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
from django.conf import settings

from ..models import Quiz, QuizQuestion, QuizAnswer
from ..utils.text_utils import strip_html_tags, clean_summary_text
from .summary_service import extract_concept_entries_from_summary

OBJECTIVE_QUIZ_TYPE = "objective"
FEEDBACK_QUIZ_TYPE = "feedback"
OBJECTIVE_SBERT_MODEL_NAME = getattr(
    settings,
    "OBJECTIVE_SBERT_MODEL_NAME",
    os.environ.get("OBJECTIVE_SBERT_MODEL_NAME", "jhgan/ko-sroberta-multitask"),
)
_objective_sbert_model = None
_objective_sbert_lock = threading.Lock()
kiwi = Kiwi()


def get_objective_sbert_model():
    """객관식 오답 선택지 생성을 위한 SBERT 모델을 지연 로딩한다."""
    global _objective_sbert_model

    if _objective_sbert_model is not None:
        return _objective_sbert_model

    with _objective_sbert_lock:
        if _objective_sbert_model is None:
            print(f"[객관식 SBERT 로딩] 모델 로딩 시작: {OBJECTIVE_SBERT_MODEL_NAME}")
            _objective_sbert_model = SentenceTransformer(OBJECTIVE_SBERT_MODEL_NAME)
            print(f"[객관식 SBERT 로딩] 모델 로딩 완료: {OBJECTIVE_SBERT_MODEL_NAME}")

    return _objective_sbert_model

def extract_keywords_with_kiwi_for_objective(text):
    """Kiwi를 사용하여 객관식 후보 명사를 추출한다."""
    text = strip_html_tags(text)

    if not text:
        return []

    stopwords = {
        "오늘", "강의", "수업", "감사", "내용", "학습", "생각", "이번", "정리",
        "요약", "문제", "핵심", "개념", "포인트", "설명", "부분", "사용", "자료",
        "화면", "이미지", "시점", "참조", "관련", "결과", "원인", "과정", "의미",
    }

    keywords = []

    try:
        analyzed = kiwi.analyze(text)

        if not analyzed:
            return []

        for token in analyzed[0][0]:
            word = (token.form or "").strip()

            if token.tag not in ["NNG", "NNP"]:
                continue

            if len(word) <= 1:
                continue

            if word in stopwords:
                continue

            if re.fullmatch(r"\d+", word):
                continue

            keywords.append(word)

    except Exception as e:
        print(f"[객관식 키워드 추출 에러] {e}")
        return []

    counts = Counter(keywords)
    return [word for word, _ in counts.most_common()]

extract_keywords_with_kiwi = extract_keywords_with_kiwi_for_objective


def find_timeline_for_keyword(summary_text, keyword):
    """요약문에서 특정 키워드와 가까운 타임라인을 찾는다."""
    keyword = (keyword or "").strip()

    if not keyword:
        return "근거 없음"

    try:
        entries = extract_concept_entries_from_summary(summary_text)

        for entry in entries:
            entry_keyword = (entry.get("keyword") or "").strip()

            if not entry_keyword:
                continue

            if keyword == entry_keyword or keyword in entry_keyword or entry_keyword in keyword:
                return entry.get("timeline") or "근거 없음"

    except Exception as e:
        print(f"[객관식 타임라인 추출 에러] {e}")

    return "근거 없음"

def generate_distractors_with_sbert(answer, all_candidates, n=3):
    """
    SBERT 임베딩을 사용하여 정답 단어와 의미적으로 가깝지만,
    품사와 규칙 검사를 통해 이상하지 않은 '매력적인 오답' 후보를 찾는다.
    """
    answer = (answer or "").strip()
    candidates = [
        str(candidate).strip()
        for candidate in all_candidates
        if str(candidate).strip()
        and str(candidate).strip() != answer
        and len(str(candidate).strip()) > 1
    ]

    # 중복 제거
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    candidates = deduped

    if not answer or not candidates:
        return []

    try:
        model = get_objective_sbert_model()
        answer_embedding = model.encode(answer, convert_to_tensor=True)
        candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
        cos_scores = util.cos_sim(answer_embedding, candidate_embeddings)[0]

        # 기존보다 더 많은 후보를 본 뒤 필터링한다.
        top_k_val = min(len(candidates), max(n + 30, 50))
        top_results = torch.topk(cos_scores, k=top_k_val)

        selected = []

        for idx in top_results.indices.tolist():
            candidate = candidates[idx]
            score = float(cos_scores[idx])

            # 1. 유사도 점수 제한
            # 너무 멀면 엉뚱한 오답이고, 너무 가까우면 정답과 거의 같은 단어일 가능성이 높다.
            if not (0.35 <= score <= 0.75):
                continue

            # 2. 문자 정규화 및 길이 검사
            if re.fullmatch(r"[\d\W_]+", candidate):
                continue

            if re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣ]+", candidate):
                continue

            if abs(len(candidate) - len(answer)) > 5:
                continue

            # 3. Kiwi 형태소 분석 기반 필터링
            try:
                tokens = kiwi.tokenize(candidate)

                if not tokens:
                    continue

                # 조사, 어미, 접사로만 이루어진 후보 제외
                invalid_tags = sum(
                    1
                    for token in tokens
                    if token.tag.startswith("J")
                    or token.tag.startswith("E")
                    or token.tag.startswith("X")
                )

                if invalid_tags == len(tokens):
                    continue

                # 의미 없는 말버릇 후보 제외
                if any(token.form in ["음", "어", "이제", "약간", "그니까"] for token in tokens):
                    continue

                # 실질적인 핵심 품사가 있는지 확인
                has_meaningful_core = any(
                    token.tag.startswith("N")
                    or token.tag.startswith("V")
                    or token.tag.startswith("XR")
                    or token.tag == "SL"
                    for token in tokens
                )

                if not has_meaningful_core:
                    continue

            except Exception:
                continue

            if candidate not in selected:
                selected.append(candidate)

            if len(selected) >= n:
                break

        # 부족하면 이번 강의 전체 후보군에서 품사가 안전한 명사 후보를 추가한다.
        if len(selected) < n:
            fallback_pool = candidates[:]
            random.shuffle(fallback_pool)

            for fallback in fallback_pool:
                if len(selected) >= n:
                    break

                if fallback == answer or fallback in selected:
                    continue

                try:
                    fallback_tokens = kiwi.tokenize(fallback)

                    if fallback_tokens and any(token.tag.startswith("N") for token in fallback_tokens):
                        selected.append(fallback)

                except Exception:
                    continue

        # 그래도 부족할 때만 최후의 고정 오답을 사용한다.
        if len(selected) < n:
            static_words = ["주요 개념", "핵심 이론", "상세 특징"]

            for static_word in static_words:
                if len(selected) >= n:
                    break

                if static_word != answer and static_word not in selected:
                    selected.append(static_word)

        return selected

    except Exception as e:
        print(f"[객관식 SBERT 오답 생성 에러] {e}")

        fallback = [
            candidate
            for candidate in candidates
            if len(candidate) > 1 and not re.fullmatch(r"[\d\W_]+", candidate)
        ]

        random.shuffle(fallback)
        return fallback[:n]


def _clean_text_for_objective(text):
    """객관식 생성용 텍스트를 정리한다."""
    text = clean_summary_text(text) if "clean_summary_text" in globals() else (text or "")
    text = strip_html_tags(text)
    text = re.sub(r"\[IMG:\s*\d{1,3}:\d{2}\]", " ", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\([^)]*?시점[^)]*?\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_objective_quiz_items(summary_text, question_count=20, previous_quiz_texts=""):
    """요약문을 기반으로 Kiwi + SBERT 객관식 문제 목록을 생성한다.

    1차로 실제 요약 문장 기반 빈칸 문제를 만들고,
    문장 수가 부족하면 핵심 키워드 기반 fallback 문제를 추가 생성한다.
    """
    clean_text = _clean_text_for_objective(summary_text)

    if not clean_text or len(clean_text) < 30:
        return []

    try:
        raw_sentences = [sent.text.strip() for sent in kiwi.split_into_sents(clean_text) if sent.text.strip()]
    except Exception:
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?。다])\s+", clean_text) if s.strip()]

    junk_words = {
        "오늘", "강의", "수업", "감사", "안녕하세요", "여러분", "요약", "정리",
        "출제", "포인트", "이미지", "시점", "자료", "참조"
    }

    filtered_sentences = []
    for sent in raw_sentences:
        sent = re.sub(r"\s+", " ", str(sent).strip())
        if len(sent) < 18:
            continue
        if any(word in sent[:15] for word in junk_words):
            continue
        if "IMG:" in sent or "판서 및 자료 참조" in sent:
            continue
        filtered_sentences.append(sent)

    if len(filtered_sentences) < 3:
        filtered_sentences = [
            re.sub(r"\s+", " ", s).strip()
            for s in raw_sentences
            if len(str(s).strip()) >= 12
        ]

    all_nouns = extract_keywords_with_kiwi_for_objective(clean_text)

    if len(all_nouns) < 4:
        return []

    used_answers = set()
    used_sentences = set()

    for match in re.finditer(r"정답\s*단어\s*:\s*([^\n]+)", previous_quiz_texts or ""):
        used_answers.add(match.group(1).strip())

    random.shuffle(filtered_sentences)
    quiz_items = []

    def make_item(answer_word, question_text, source_sentence=""):
        distractors = generate_distractors_with_sbert(answer_word, all_nouns, n=3)

        if len(distractors) < 3:
            remaining = [
                noun for noun in all_nouns
                if noun != answer_word and noun not in distractors and noun not in used_answers
            ]
            random.shuffle(remaining)
            distractors.extend(remaining[:3 - len(distractors)])

        if len(distractors) < 3:
            return None

        options = distractors[:3] + [answer_word]
        random.shuffle(options)
        correct_choice = str(options.index(answer_word) + 1)
        timeline = find_timeline_for_keyword(summary_text, answer_word)
        explanation = f"본 문항은 강의 요약의 핵심 키워드 '{answer_word}'를 확인하기 위한 문제입니다."

        return {
            "number": len(quiz_items) + 1,
            "question_text": question_text,
            "choices": [
                {"c_num": str(index + 1), "choice_text": option}
                for index, option in enumerate(options)
            ],
            "correct_choice": correct_choice,
            "answer_word": answer_word,
            "keyword": answer_word,
            "timeline": timeline,
            "explanation": explanation,
            "source_sentence": source_sentence,
        }

    # 1차: 실제 문장 기반 문제
    for sent in filtered_sentences:
        if len(quiz_items) >= question_count:
            break
        if sent in used_sentences:
            continue

        sent_nouns = extract_keywords_with_kiwi_for_objective(sent)
        if not sent_nouns:
            continue

        answer_word = None
        for noun in sent_nouns:
            if noun in used_answers:
                continue
            if noun in all_nouns[:60]:
                answer_word = noun
                break

        if not answer_word:
            continue

        question_sentence = sent.replace(answer_word, "( ____ )", 1)
        item = make_item(
            answer_word=answer_word,
            question_text=f"다음 빈칸에 알맞은 단어를 고르세요.\n{question_sentence}",
            source_sentence=sent,
        )

        if not item:
            continue

        quiz_items.append(item)
        used_sentences.add(sent)
        used_answers.add(answer_word)

    # 2차: 핵심 키워드 fallback 문제
    fallback_keywords = [
        noun for noun in all_nouns
        if noun not in used_answers and len(noun) > 1
    ]

    for keyword in fallback_keywords:
        if len(quiz_items) >= question_count:
            break

        item = make_item(
            answer_word=keyword,
            question_text=(
                "강의에서 설명한 핵심 개념 중 다음 빈칸에 알맞은 단어를 고르세요.\n"
                "강의에서 중요한 개념으로 다루어진 ( ____ )에 대한 설명을 이해해야 한다."
            ),
            source_sentence="",
        )

        if not item:
            continue

        quiz_items.append(item)
        used_answers.add(keyword)

    return quiz_items

def objective_items_to_text(items):
    """객관식 문제 목록을 사람이 읽을 수 있는 원본 텍스트로 변환한다."""
    if not items:
        return "퀴즈를 생성할 충분한 핵심 내용이 없습니다."

    marks = ["①", "②", "③", "④"]
    blocks = []

    for item in items:
        option_text = "  ".join([
            f"{marks[index]} {choice.get('choice_text', '')}"
            for index, choice in enumerate(item.get("choices") or [])
        ])
        block = f"""{item.get('number')}. 문제 : {item.get('question_text')}
{option_text}
모범 답안 : {item.get('correct_choice')}
정답 단어 : {item.get('answer_word')}
관련 키워드 : {item.get('keyword')}
관련 타임라인 : {item.get('timeline')}
해설 : {item.get('explanation')}"""
        blocks.append(block.strip())

    return "\n\n".join(blocks)

def generate_objective_quiz(summary_text, question_count=20, previous_quiz_texts=""):
    """요약문 기반 객관식 문제 원본 텍스트를 생성한다."""
    items = generate_objective_quiz_items(
        summary_text=summary_text,
        question_count=question_count,
        previous_quiz_texts=previous_quiz_texts,
    )
    return objective_items_to_text(items)

def parse_objective_quiz_text(quiz_text):
    """객관식 원본 텍스트를 구조화한다."""
    quiz_text = quiz_text or ""
    items = []

    blocks = re.split(r"(?m)^\s*(?=\d+\.\s*문제\s*:)", quiz_text.strip())

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        number_match = re.match(r"(\d+)\.\s*문제\s*:\s*", block)

        if not number_match:
            continue

        number = int(number_match.group(1))
        rest = block[number_match.end():].strip()

        answer_match = re.search(r"(?m)^\s*모범\s*답안\s*:\s*([^\n]+)", rest)
        answer_word_match = re.search(r"(?m)^\s*정답\s*단어\s*:\s*([^\n]+)", rest)
        keyword_match = re.search(r"(?m)^\s*관련\s*키워드\s*:\s*([^\n]+)", rest)
        timeline_match = re.search(r"(?m)^\s*관련\s*타임라인\s*:\s*([^\n]+)", rest)
        explanation_match = re.search(r"(?m)^\s*해설\s*:\s*([^\n]+)", rest)

        correct_choice = answer_match.group(1).strip() if answer_match else ""
        answer_word = answer_word_match.group(1).strip() if answer_word_match else ""
        keyword = keyword_match.group(1).strip() if keyword_match else answer_word
        timeline = timeline_match.group(1).strip() if timeline_match else "근거 없음"
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        question_part = rest
        for marker in ["모범 답안", "정답 단어", "관련 키워드", "관련 타임라인", "해설"]:
            idx = question_part.find(marker)
            if idx >= 0:
                question_part = question_part[:idx].strip()
                break

        choice_symbols = ["①", "②", "③", "④"]
        symbol_positions = []

        for symbol in choice_symbols:
            pos = question_part.find(symbol)
            if pos >= 0:
                symbol_positions.append((symbol, pos))

        symbol_positions.sort(key=lambda x: x[1])

        choices = []

        if symbol_positions:
            question_text = question_part[:symbol_positions[0][1]].strip()
            choices_text = question_part[symbol_positions[0][1]:]

            for idx, symbol in enumerate(choice_symbols, start=1):
                pattern = rf"{re.escape(symbol)}\s*(.*?)(?={'|'.join(map(re.escape, choice_symbols))}|$)"
                match = re.search(pattern, choices_text, re.DOTALL)
                choice_text = match.group(1).strip() if match else f"{idx}번 선택지"
                choices.append({"c_num": str(idx), "choice_text": choice_text})
        else:
            question_text = question_part.strip()
            choices = [
                {"c_num": str(idx), "choice_text": f"{idx}번 선택지"}
                for idx in range(1, 5)
            ]

        items.append({
            "number": number,
            "question_text": question_text,
            "choices": choices,
            "correct_choice": str(correct_choice).strip(),
            "answer_word": answer_word,
            "keyword": keyword,
            "timeline": timeline,
            "explanation": explanation,
        })

    return items

def save_objective_questions_from_text(quiz, quiz_text):
    """객관식 quiz_text를 QuizQuestion에 저장한다."""
    items = parse_objective_quiz_text(quiz_text)

    if not items:
        print("[객관식 저장] 파싱된 문제가 없어 저장을 건너뜁니다.")
        return

    QuizQuestion.objects.filter(quiz=quiz).delete()

    for item in items:
        meta = {
            "choices": item.get("choices") or [],
            "correct_choice": str(item.get("correct_choice") or ""),
            "answer_word": item.get("answer_word") or "",
            "keyword": item.get("keyword") or "",
            "explanation": item.get("explanation") or "",
        }

        QuizQuestion.objects.create(
            quiz=quiz,
            number=item.get("number") or 1,
            question_text=item.get("question_text") or "",
            model_answer=str(item.get("correct_choice") or ""),
            explanation=json.dumps(meta, ensure_ascii=False),
            related_timeline=(item.get("timeline") or "")[:500],
        )

    print(f"[객관식 저장] {quiz.generation_number}차 문제 {len(items)}개를 저장했습니다.")

def ensure_objective_questions_exist(quiz):
    """객관식 QuizQuestion이 없으면 quiz_text를 파싱해 저장한다."""
    if not quiz.questions.exists() and quiz.quiz_text:
        save_objective_questions_from_text(quiz, quiz.quiz_text)

def get_objective_meta(question):
    """QuizQuestion.explanation에 저장된 객관식 JSON 메타데이터를 읽는다."""
    raw = question.explanation or ""

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    return {
        "choices": [],
        "correct_choice": question.model_answer or "",
        "answer_word": "",
        "keyword": "",
        "explanation": raw,
    }

def build_objective_items_for_display(quiz, user=None):
    """객관식 퀴즈 화면/채점 화면에 사용할 데이터를 구성한다."""
    ensure_objective_questions_exist(quiz)

    items = []

    for question in quiz.questions.order_by("number", "id"):
        meta = get_objective_meta(question)
        answer_obj = None

        if user and user.is_authenticated:
            answer_obj = QuizAnswer.objects.filter(
                user=user,
                quiz_question=question,
            ).first()

        user_choice = answer_obj.user_answer if answer_obj else ""
        correct_choice = str(meta.get("correct_choice") or question.model_answer or "").strip()
        choices = meta.get("choices") or []

        # c_num을 문자열로 통일한다.
        normalized_choices = []
        for idx, choice in enumerate(choices, start=1):
            normalized_choices.append({
                "c_num": str(choice.get("c_num") or idx),
                "choice_text": choice.get("choice_text") or "",
            })

        while len(normalized_choices) < 4:
            next_num = len(normalized_choices) + 1
            normalized_choices.append({
                "c_num": str(next_num),
                "choice_text": f"{next_num}번 선택지",
            })

        choice_items = normalized_choices[:4]
        option_map = {
            str(choice["c_num"]): choice["choice_text"]
            for choice in choice_items
        }

        user_choice_text = str(user_choice or "").strip()

        items.append({
            # 기존 템플릿 호환용 필드
            "q_id": question.id,
            "q_num": question.number,
            "number": question.number,
            "question_text": question.question_text,
            "choices": choice_items,
            "correct_choice": correct_choice,
            "user_choice": user_choice_text,
            "answer_word": meta.get("answer_word") or "",
            "keyword": meta.get("keyword") or meta.get("answer_word") or "",
            "timeline": question.related_timeline or "",
            "explanation": meta.get("explanation") or "",

            # 팀원 객관식 UI 템플릿 호환용 필드
            "id": question.id,
            "user_answer": user_choice_text,
            "options": option_map,
        })

    return items

def get_objective_quizzes_for_lecture(lecture):
    """해당 강의의 객관식 퀴즈 차수만 조회한다."""
    return (
        Quiz.objects
        .filter(lecture=lecture, explanation=OBJECTIVE_QUIZ_TYPE)
        .exclude(quiz_text="")
        .order_by("generation_number", "id")
    )
