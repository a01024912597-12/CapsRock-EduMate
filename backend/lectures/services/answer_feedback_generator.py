"""학생 답안의 피드백을 생성하며, 키워드 일치만으로 정답을 판단하지 않는다."""
import re
import hashlib
import json

from .grading_service import extract_grading_keywords
from ..utils.quiz_language import LANG_KO, LANG_EN, LANG_ZH, detect_quiz_language


STATUS_LABELS = {
    "met": "충족", "partial": "부분 충족", "missing": "미충족",
}
ERROR_TYPES = {"reversed_relation", "cause_effect", "concept_confusion", "incorrect_fact"}
ERROR_LABELS = {"reversed_relation": "관계·방향 오류", "cause_effect": "인과관계 혼동",
                "concept_confusion": "개념 혼동", "incorrect_fact": "사실관계 오류"}



def _learning_language(*texts):
    return detect_quiz_language("\n".join(str(text or "") for text in texts))


def _status_labels(language):
    if language == LANG_EN:
        return {"met": "Met", "partial": "Partially met", "missing": "Missing"}
    if language == LANG_ZH:
        return {"met": "已满足", "partial": "部分满足", "missing": "未满足"}
    return STATUS_LABELS


def _error_labels(language):
    if language == LANG_EN:
        return {"reversed_relation": "Reversed relation", "cause_effect": "Cause/effect confusion",
                "concept_confusion": "Concept confusion", "incorrect_fact": "Factual error"}
    if language == LANG_ZH:
        return {"reversed_relation": "关系方向错误", "cause_effect": "因果关系混淆",
                "concept_confusion": "概念混淆", "incorrect_fact": "事实错误"}
    return ERROR_LABELS


def _learning_text(language, key, **kwargs):
    table = {
        LANG_KO: {
            "criterion": "핵심 설명 {n}",
            "fallback_hint": "‘{reference}’에서 설명한 내용을 자신의 말로 정리해 보세요.",
            "direction_hint": "‘{keyword}’와 관련해 모범 답안에서 설명한 변화 방향을 확인해 보세요.",
            "cause_hint": "‘{keyword}’가 문제에서 묻는 원인이나 이유와 어떻게 연결되는지 설명해 보세요.",
            "effect_hint": "‘{keyword}’가 어떤 역할을 하거나 어떤 결과에 연결되는지 설명해 보세요.",
            "concept_hint": "‘{keyword}’의 의미를 설명하고 문제에서 묻는 개념과 연결해 보세요.",
            "unverified": "다른 표현으로 설명했을 수 있어 규칙만으로 정확성을 판단하지 못했어요. 모범 답안과 의미를 비교해 보세요.",
            "met": "모범 답안의 해당 설명과 일치하는 내용을 확인했어요.",
            "reversed": "모범 답안과 작용 방향이 반대로 표현되어 있어요. 변화 방향을 확인해 보세요.",
            "missing_terms": "모범 답안에서 추출한 핵심 용어의 표현을 확인하지 못했어요. 같은 의미로 설명했는지 점검해 보세요.",
            "partial_terms": "필수 개념의 표현은 확인했어요. 개념 사이의 관계까지 정확한지는 모범 답안과 비교해 보세요.",
            "matched_terms": "'{terms}' 표현을 확인했어요. 용어를 언급하는 것뿐 아니라 서로의 관계까지 설명했는지 확인해 보세요.",
            "direction_compare": "모범 답안과 내 답안에서 증가·감소 또는 높아짐·낮아짐의 방향을 비교해 보세요.",
            "relation_compare": "‘{reference}’에서 설명한 관계를 내 답안에도 표현했는지 확인해 보세요.",
            "sentence_unverified": "표현 일치만으로 이 문장의 정확성을 판단하기 어려워요. 문제에서 요구한 설명이 들어 있는지 확인해 보세요.",
            "strength": "모범 답안과 일치하는 핵심 설명을 확인했어요.",
            "general_hint": "핵심 개념을 다른 상황의 예시에도 적용할 수 있는지 설명해 보세요.",
            "empty_hint": "문제에서 요구한 개념의 의미를 한 문장으로 작성해 보세요.",
            "no_rubric": "평가 기준이 없어 상세 판단을 보류했어요. 모범 답안을 확인해 보세요.",
            "default_term_hint": "'{keyword}'의 의미와 문제에서 요구한 관계를 설명해 보세요.",
            "default_retry_hint": "문제에서 요구한 개념과 설명 사이의 관계를 한 문장으로 다시 정리해보세요.",
            "notice": "모범 답안에서 자동으로 추출한 핵심 용어와 설명을 비교한 피드백이에요. 같은 의미의 다른 표현을 모두 인식하지는 못할 수 있어요.",
            "remaining_reversed": "모범 답안의 ‘{reference}’와 비교해, 내 답안에서 반대로 표현한 작용 방향을 수정해 보세요.",
            "remaining_unverified": "‘{reference}’와 같은 의미를 내 답안에서도 설명했는지 확인해 보세요. 다른 표현으로 설명했다면 의미가 일치하는지 비교해 보세요.",
            "remaining_relation": "‘{reference}’에서 설명하는 개념 사이의 관계가 드러나도록 내 답안을 보완해 보세요.",
            "default_label": "핵심 설명",
            "default_error": "개념 오류",
        },
        LANG_EN: {
            "criterion": "Key explanation {n}",
            "fallback_hint": "Restate the idea described in '{reference}' in your own words.",
            "direction_hint": "Check the direction of change described for '{keyword}' in the model answer.",
            "cause_hint": "Explain how '{keyword}' connects to the cause or reason asked in the question.",
            "effect_hint": "Explain what role '{keyword}' plays or what result it leads to.",
            "concept_hint": "Explain the meaning of '{keyword}' and connect it to what the question asks.",
            "unverified": "The wording may differ, so the rule-based check cannot verify it confidently. Compare the meaning with the model answer.",
            "met": "This part matches the corresponding explanation in the model answer.",
            "reversed": "The direction of the relationship is reversed compared with the model answer. Check the direction of change.",
            "missing_terms": "The key terms extracted from the model answer were not found. Check whether you expressed the same meaning in different words.",
            "partial_terms": "The required concept is present. Compare the relationship between concepts with the model answer.",
            "matched_terms": "The terms '{terms}' were found. Make sure the relationships between them are also explained.",
            "direction_compare": "Compare the direction of increase/decrease or raise/lower with the model answer.",
            "relation_compare": "Check whether your answer expresses the relationship described in '{reference}'.",
            "sentence_unverified": "Matching wording alone is not enough to verify this sentence. Check whether it answers what the question asks.",
            "strength": "A key explanation matching the model answer was found.",
            "general_hint": "Try applying the key concept to another example or situation.",
            "empty_hint": "Write one sentence explaining the concept asked in the question.",
            "no_rubric": "Detailed judgment was withheld because no evaluation rubric is available. Review the model answer.",
            "default_term_hint": "Explain the meaning of '{keyword}' and its relationship to what the question asks.",
            "default_retry_hint": "Restate the relationship between the concept and explanation in one sentence.",
            "notice": "This feedback compares automatically extracted key terms and explanations with the model answer. It may not recognize every equivalent expression.",
            "remaining_reversed": "Compare with '{reference}' and correct the reversed direction in your answer.",
            "remaining_unverified": "Check whether your answer expresses the same meaning as '{reference}', even if you used different wording.",
            "remaining_relation": "Revise your answer so the relationship between the concepts in '{reference}' is clear.",
            "default_label": "Key explanation",
            "default_error": "Concept error",
        },
        LANG_ZH: {
            "criterion": "核心说明 {n}",
            "fallback_hint": "请用自己的话重新说明“{reference}”中的内容。",
            "direction_hint": "请确认标准答案中“{keyword}”相关变化的方向。",
            "cause_hint": "请说明“{keyword}”与题目所问原因之间的联系。",
            "effect_hint": "请说明“{keyword}”发挥什么作用或会带来什么结果。",
            "concept_hint": "请说明“{keyword}”的含义，并与题目要求联系起来。",
            "unverified": "可能使用了不同表达，因此规则无法准确判断。请与标准答案比较含义。",
            "met": "已确认该部分与标准答案中的对应说明一致。",
            "reversed": "与标准答案相比，关系方向表达相反。请确认变化方向。",
            "missing_terms": "未找到从标准答案提取的核心词。请检查是否用其他表达说明了相同含义。",
            "partial_terms": "已找到必要概念，但还需与标准答案比较概念之间的关系。",
            "matched_terms": "已找到“{terms}”等表达。除了术语本身，也请确认是否说明了它们之间的关系。",
            "direction_compare": "请比较标准答案和你的答案中增加/减少、提高/降低的方向。",
            "relation_compare": "请确认你的答案是否表达了“{reference}”中说明的关系。",
            "sentence_unverified": "仅凭表述一致无法判断该句是否准确。请确认是否包含题目要求的说明。",
            "strength": "已确认与标准答案一致的核心说明。",
            "general_hint": "请尝试把核心概念应用到另一个例子或情境中。",
            "empty_hint": "请用一句话说明题目要求的概念含义。",
            "no_rubric": "由于没有评价标准，暂不进行详细判断。请参考标准答案。",
            "default_term_hint": "请说明“{keyword}”的含义及其与题目要求之间的关系。",
            "default_retry_hint": "请用一句话重新整理题目要求的概念与说明之间的关系。",
            "notice": "本反馈将从标准答案中自动提取的核心词和说明与答题内容进行比较，可能无法识别所有同义表达。",
            "remaining_reversed": "请与“{reference}”比较，并修正答案中表达相反的作用方向。",
            "remaining_unverified": "请确认你的答案是否表达了与“{reference}”相同的含义，即使使用了不同措辞。",
            "remaining_relation": "请补充答案，使“{reference}”中概念之间的关系更加明确。",
            "default_label": "核心说明",
            "default_error": "概念错误",
        },
    }
    return table.get(language, table[LANG_KO]).get(key, "").format(**kwargs)


def build_feedback_rubric(question_text, model_answer):
    """재제출 시에도 같은 기준으로 평가하도록 모범 답안의 핵심 설명을 고정한다."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+|\n+", model_answer or "") if s.strip()]
    rubric = [
        {"id": f"criterion_{i}", "label": _learning_text(_learning_language(question_text, model_answer), "criterion", n=i), "reference": sentence}
        for i, sentence in enumerate(sentences[:4], 1)
    ]
    return prepare_feedback_rubric(question_text, model_answer, rubric)


def prepare_feedback_rubric(question_text, model_answer, rubric):
    """문제와 모범 답안에서 핵심 용어, 인정 표현, 수정 힌트를 자동으로 준비한다."""
    references = _sentences(model_answer)[:4]
    if [row.get("reference") for row in rubric] != references:
        language = _learning_language(question_text, model_answer)
        rubric = [{"id": f"criterion_{index}", "label": _learning_text(language, "criterion", n=index), "reference": reference}
                  for index, reference in enumerate(references, 1)]
    language = _learning_language(question_text, model_answer)
    result = []
    question_terms = {_normalize(term) for term in _content_terms(question_text)}
    source_signature = hashlib.sha256((question_text + "\n" + model_answer).encode("utf-8")).hexdigest()
    for original in rubric:
        row = dict(original)
        reference = row["reference"]
        if row.get("rules_version") == 2 and row.get("rules_source_signature") == source_signature:
            result.append(row)
            continue
        # 질문에 이미 나온 개념명을 다시 쓰도록 강요하지 않고, 설명에 필요한 용어를 우선한다.
        terms = [term for term in _content_terms(reference) if _normalize(term) not in question_terms]
        terms.sort(key=lambda term: not bool(re.search(re.escape(term) + r"(?:을|를|이|가)(?:\s|$)", reference)))
        required = []
        for term in terms[:2]:
            aliases = []
            match = re.search(re.escape(term) + r"\s*\(([가-힣A-Za-z][가-힣A-Za-z0-9 -]{1,39})\)", reference)
            if match:
                aliases.append(match.group(1).strip())
            # 한국어 복합 용어의 띄어쓰기 차이는 별도 의미 추측 없이 인정할 수 있다.
            compact = re.sub(r"\s+", "", term)
            if compact != term and re.fullmatch(r"[가-힣]+", compact):
                aliases.append(compact)
            required.append({"keyword": term, "aliases": list(dict.fromkeys(aliases)),
                             "hint": _concept_hint(question_text, reference, term, language), "source": "automatic"})
        row.pop("candidate_terms", None)
        row["required_terms"] = required
        row["hint"] = (_concept_hint(question_text, reference, terms[0], language) if terms
                       else _learning_text(language, "fallback_hint", reference=reference))
        row["rules_version"] = 2
        row["rules_source_signature"] = source_signature
        result.append(row)
    return result


def _content_terms(text):
    """일반어·영문 기능어를 제외하고 원문에 근거한 복합 용어를 우선 추출한다."""
    generic = {"내용", "설명", "개념", "특징", "의미", "역할", "이유", "중요", "핵심", "정도", "사용", "이해",
               "강의", "문제", "답안", "해당", "관련", "경우", "전체", "하나", "정리", "상태", "현상",
               "서술", "때문", "대해", "통해", "위해", "대한", "the", "a", "an", "is", "are", "was", "were",
               "of", "to", "for", "in", "on", "and", "or", "it", "its", "this", "that", "with", "by", "as",
               "explain", "describe", "define", "discuss", "what", "why", "how", "does", "do", "has", "have", "can", "be", "been",
               "produce", "produces", "produced", "generate", "generates", "lowers", "raises", "increases", "decreases"}
    if any(phrase in text for phrase in ("답안에는", "강의에서 다룬 핵심 개념", "해당 개념의 의미")):
        return []
    raw = [*extract_grading_keywords(text), *re.findall(r"[A-Za-z][A-Za-z0-9-]+", text)]
    terms, seen = [], set()
    for value in raw:
        value = value.strip(".,!?() ")
        key = _normalize(value)
        if not 2 <= len(value) <= 40 or key in generic or key in seen or not _contains(text, value):
            continue
        if re.search(r"\(\s*" + re.escape(value) + r"\s*\)", text, re.IGNORECASE):
            continue
        seen.add(key)
        terms.append(value)
    # 연속된 명사 표현을 합쳐 '일반화 성능' 같은 하나의 개념이 조각나지 않게 한다.
    terms.sort(key=lambda term: text.casefold().find(term.casefold()))
    phrases = []
    for term in terms:
        if phrases:
            previous = phrases[-1]
            start = text.casefold().find(previous.casefold())
            end = start + len(previous)
            next_start = text.casefold().find(term.casefold(), end)
            between = text[end:next_start] if next_start >= end else None
            korean_phrase = bool(re.search(r"[가-힣]", previous) and re.search(r"[가-힣]", term))
            if (korean_phrase and between is not None and (not between or between.isspace())
                    and len(previous) + len(term) + 1 <= 40):
                phrases[-1] = text[start:next_start + len(term)]
                continue
        phrases.append(term)
    return phrases


def _concept_hint(question_text, reference, keyword, language=None):
    language = language or _learning_language(question_text, reference)
    if any(word in reference for word in ("낮추", "높이", "증가", "감소", "상승", "하락", "lowers", "raises", "increases", "decreases", "提高", "降低", "增加", "减少")):
        return _learning_text(language, "direction_hint", keyword=keyword)
    if any(word.casefold() in question_text.casefold() for word in ("원인", "이유", "배경", "cause", "reason", "why", "原因", "理由", "背景")):
        return _learning_text(language, "cause_hint", keyword=keyword)
    if any(word.casefold() in question_text.casefold() for word in ("영향", "결과", "역할", "effect", "result", "role", "影响", "结果", "作用")):
        return _learning_text(language, "effect_hint", keyword=keyword)
    return _learning_text(language, "concept_hint", keyword=keyword)


def feedback_rule_signature(rubric):
    """규칙을 수정한 뒤 같은 답안을 다시 제출해도 새 규칙으로 평가하게 한다."""
    return hashlib.sha256(json.dumps(rubric, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?。])\s+|\n+", text or "") if s.strip()]


def polite_hint(text):
    """기존 힌트에 남아 있는 명령형 표현도 정중한 안내 표현으로 바꾼다."""
    return re.sub(r"해\s*보라(?=[.!?。]?(?:\s|$))", "해 보세요", text)


def build_learning_feedback(question, user_answer):
    """외부 생성형 인공지능 호출 없이 문장 비교와 자동 추출한 용어로 피드백을 만든다."""
    language = _learning_language(question.question_text, question.model_answer, user_answer)
    rubric = prepare_feedback_rubric(question.question_text, question.model_answer, question.feedback_rubric) if question.feedback_rubric else build_feedback_rubric(question.question_text, question.model_answer)
    sentences = _sentences(user_answer)
    data = {"language": language, "criteria": [], "sentence_feedback": [], "strengths": [], "hints": [],
            "misconceptions": [], "missing_required_keywords": []}
    for criterion in rubric:
        reference = criterion["reference"]
        exact = next((sentence for sentence in sentences if _normalize(sentence) == _normalize(reference)), "")
        opposite = next((sentence for sentence in sentences if _is_reversed(reference, sentence)), "")
        required = _required_terms(criterion)
        # 단어 일치는 내용 확인을 돕는 근거로만 사용하며 정답 판정 근거로 쓰지 않는다.
        reference_terms = extract_grading_keywords(reference)[:6]
        related = max(sentences, key=lambda sentence: sum(_contains(sentence, term) for term in reference_terms), default="")
        matched_terms = [term for term in reference_terms if related and _contains(related, term)]
        missing = [term for term in required if not any(_contains(user_answer, expression)
                   for expression in [term["keyword"], *term["aliases"]])]
        status, evidence = "unverified", ""
        comment = _learning_text(language, "unverified")
        if exact and not opposite:
            status, evidence = "met", exact
            comment = _learning_text(language, "met")
        elif opposite:
            status, evidence = "error", opposite
            comment = _learning_text(language, "reversed")
            data["misconceptions"].append({"criterion_id": criterion["id"], "type": "reversed_relation",
                                           "evidence": opposite, "message": comment})
        elif required and missing:
            status = "partial" if len(missing) < len(required) else "missing"
            if status == "partial":
                evidence = next(sentence for sentence in sentences if any(
                    _contains(sentence, expression) for term in required for expression in [term["keyword"], *term["aliases"]]
                ))
            comment = _learning_text(language, "missing_terms")
        elif required and sentences:
            status = "partial"
            evidence = next((sentence for sentence in sentences if any(
                _contains(sentence, expression) for term in required for expression in [term["keyword"], *term["aliases"]]
            )), sentences[0])
            comment = _learning_text(language, "partial_terms")
        elif matched_terms:
            status, evidence = "partial", related
            comment = _learning_text(language, "matched_terms", terms=", ".join(matched_terms[:3]))
        data["criteria"].append({"id": criterion["id"], "status": status, "evidence": evidence, "comment": comment})
        if status in {"missing", "partial"}:
            for term in missing:
                data["missing_required_keywords"].append({"criterion_id": criterion["id"],
                    "keyword": term["keyword"], "reason": term["hint"]})
                data["hints"].append(term["hint"])
        elif status == "error":
            data["hints"].append(_learning_text(language, "direction_compare"))
        elif status != "met":
            data["hints"].append(criterion.get("hint") or _learning_text(language, "relation_compare", reference=reference))
    for sentence in sentences:
        judgments = [row for row in data["criteria"] if row["evidence"] == sentence]
        row = next((row for row in judgments if row["status"] == "error"), None)
        row = row or next((row for row in judgments if row["status"] == "met"), None)
        row = row or next((row for row in judgments if row["status"] == "partial"), None)
        data["sentence_feedback"].append({"text": sentence, "status": row["status"] if row else "unverified",
            "comment": row["comment"] if row else _learning_text(language, "sentence_unverified")})
    if any(row["status"] == "met" for row in data["criteria"]):
        data["strengths"] = [_learning_text(language, "strength")]
    data["hints"] = list(dict.fromkeys(data["hints"]))[:2]
    if not data["hints"]:
        data["hints"] = [_learning_text(language, "general_hint") if sentences
                         else _learning_text(language, "empty_hint")]
    result = validate_learning_feedback(data, rubric, user_answer)
    result["rule_signature"] = feedback_rule_signature(rubric)
    if not rubric:
        result["status"] = "unavailable"
        result["notice"] = _learning_text(language, "no_rubric")
    return result


def _normalize(text):
    text = re.sub(r"[.!?。,]", "", text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", text)


def _contains(text, expression):
    """영문은 단어 경계를 확인하고 한국어는 띄어쓰기 차이를 허용한다."""
    if re.fullmatch(r"[A-Za-z0-9 ]+", expression):
        return bool(re.search(r"(?<!\w)" + re.escape(expression) + r"(?!\w)", text, re.IGNORECASE))
    return _normalize(expression) in _normalize(text)


def _required_terms(criterion):
    """자동으로 준비된 핵심 용어와 인정 표현, 수정 힌트를 읽는다."""
    terms = []
    for term in criterion.get("required_terms", []):
        if isinstance(term, str):
            term = {"keyword": term}
        if not isinstance(term, dict) or not isinstance(term.get("keyword"), str):
            continue
        keyword = term["keyword"].strip()
        if not 2 <= len(keyword) <= 40 or not _contains(criterion["reference"], keyword):
            continue
        aliases = term.get("aliases", [])
        aliases = [alias for alias in aliases if isinstance(alias, str) and alias.strip()] if isinstance(aliases, list) else []
        hint = term.get("hint")
        if not isinstance(hint, str) or not hint.strip():
            hint = _learning_text(_learning_language(criterion.get("reference", "")), "default_term_hint", keyword=keyword)
        terms.append({"keyword": keyword, "aliases": aliases, "hint": polite_hint(hint)})
    return terms


def _is_reversed(reference, sentence):
    """나머지 표현이 동일하고 방향 표현 하나만 반대인 경우에만 오류를 확정한다."""
    original, actual = _normalize(reference), _normalize(sentence)
    for left, right in [("낮추", "높이"), ("증가", "감소"), ("상승", "하락"),
                        ("lowers", "raises"), ("increases", "decreases"),
                        ("提高", "降低"), ("增加", "减少"), ("上升", "下降")]:
        for before, after in [(left, right), (right, left)]:
            if original.count(before) == 1 and original.replace(before, after, 1) == actual:
                return True
    return False


def validate_learning_feedback(data, rubric, user_answer):
    language = data.get("language") if isinstance(data, dict) else LANG_KO
    language = language if language in {LANG_KO, LANG_EN, LANG_ZH} else LANG_KO
    labels = _status_labels(language)
    if not isinstance(data, dict):
        raise ValueError("Feedback must be an object")
    rows = data.get("criteria")
    if not isinstance(rows, list) or len(rows) != len(rubric):
        raise ValueError("Incomplete criteria")
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {row["id"] for row in rubric}:
        raise ValueError("Unknown criteria")
    criteria = []
    for criterion in rubric:
        row = by_id[criterion["id"]]
        status = row["status"]
        evidence = row.get("evidence", "")
        if status not in {*STATUS_LABELS, "error", "unverified"} or not isinstance(evidence, str):
            raise ValueError("Invalid status or evidence")
        if evidence and evidence not in user_answer:
            raise ValueError("Invented evidence")
        if status in {"met", "partial", "error"} and not evidence.strip():
            raise ValueError("Judgment requires evidence")
        comment = row.get("comment", "")
        if not isinstance(comment, str):
            raise ValueError("Invalid comment")
        label = row.get("label", criterion["label"])
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Invalid criterion label")
        criteria.append({**criterion, "label": label[:80], "status": status, "status_label": labels.get(status, labels["missing"]),
                         "evidence": evidence, "comment": comment[:800]})
    sentences = _sentences(user_answer)
    sentence_rows = data.get("sentence_feedback", [])
    if not isinstance(sentence_rows, list) or len(sentence_rows) != len(sentences):
        raise ValueError("Incomplete sentence feedback")
    sentence_feedback = []
    for text, row in zip(sentences, sentence_rows):
        if row.get("text") != text or row.get("status") not in {*STATUS_LABELS, "error", "unverified"} or not isinstance(row.get("comment"), str):
            raise ValueError("Invented sentence")
        sentence_feedback.append({"text": text, "status": row["status"],
                                  "status_label": labels.get(row["status"], labels["missing"]), "comment": row["comment"][:800]})
    misconceptions = []
    for row in data.get("misconceptions", []):
        criterion = next((c for c in criteria if c["id"] == row.get("criterion_id") and c["status"] == "error"), None)
        evidence = row.get("evidence", "")
        if (criterion and row.get("type") in ERROR_TYPES and isinstance(evidence, str)
                and evidence.strip() and evidence in user_answer and isinstance(row.get("message"), str)):
            misconceptions.append({"key": criterion["id"] + ":" + row["type"],
                                   "criterion_id": criterion["id"], "type": row["type"],
                                   "evidence": evidence, "message": row["message"][:800]})
    def text_list(key, limit):
        values = data.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("Invalid feedback list")
        return [v[:500] for v in values if v.strip()][:limit]
    hints = [polite_hint(hint) for hint in text_list("hints", 2)]
    if not hints:
        hints = [_learning_text(language, "default_retry_hint")]
    missing_keywords = validate_missing_required_keywords(data.get("missing_required_keywords", []), criteria, user_answer)
    return normalize_feedback_statuses({"version": 4, "engine": "rules", "language": language, "status": "ready", "notice": _learning_text(language, "notice"),
            "criteria": criteria, "sentence_feedback": sentence_feedback,
            "strengths": text_list("strengths", 1), "hints": hints,
            "misconceptions": misconceptions, "missing_required_keywords": missing_keywords})


def normalize_feedback_statuses(feedback):
    """이전 기록을 포함해 공개 평가 상태는 세 가지로 통일하고 언어별 라벨을 적용한다."""
    result = dict(feedback)
    language = result.get("language", LANG_KO)
    language = language if language in {LANG_KO, LANG_EN, LANG_ZH} else LANG_KO
    labels = _status_labels(language)
    for field in ("criteria", "sentence_feedback"):
        result[field] = []
        for original in feedback.get(field, []):
            row = dict(original)
            if row.get("status") == "error":
                row["assessment_reason"] = "direction_error"
            elif row.get("status") == "unverified":
                row["assessment_reason"] = "not_confirmed"
            if row.get("status") not in STATUS_LABELS:
                row["status"] = "missing"
            row["status_label"] = labels[row["status"]]
            result[field].append(row)
    return result


def validate_missing_required_keywords(rows, criteria, user_answer):
    """미충족 기준에 근거한 필수 용어만 남기고 일반어와 이미 언급한 용어를 제외한다."""
    if not isinstance(rows, list):
        return []
    generic_terms = {"내용", "설명", "특징", "의미", "개념", "역할", "강의", "문제", "답안", "핵심", "중요", "이유", "경우", "관련", "해당",
                     "content", "explanation", "feature", "meaning", "concept", "role", "lecture", "question", "answer", "key", "important", "reason",
                     "内容", "说明", "特点", "含义", "概念", "作用", "课程", "问题", "答案", "核心", "重要", "原因"}
    normalize = lambda text: re.sub(r"\s+", "", text).casefold()
    normalized_answer = normalize(user_answer)
    by_id = {c["id"]: c for c in criteria}
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        keyword, reason = row.get("keyword"), row.get("reason")
        criterion = by_id.get(row.get("criterion_id")) if isinstance(row.get("criterion_id"), str) else None
        if not isinstance(keyword, str) or not isinstance(reason, str) or not criterion:
            continue
        keyword, reason = keyword.strip(), reason.strip()
        key = normalize(keyword)
        if (criterion["status"] not in {"partial", "missing"} or not 2 <= len(keyword) <= 40
                or key in generic_terms or not reason or key in seen
                or key not in normalize(criterion["reference"]) or key in normalized_answer):
            continue
        seen.add(key)
        result.append({"keyword": keyword, "reason": reason[:500], "criterion_id": criterion["id"]})
        if len(result) == 3:
            break
    return result


def compare_attempts(previous, current):
    previous, current = normalize_feedback_statuses(previous), normalize_feedback_statuses(current)
    language = current.get("language", LANG_KO)
    old = {c["id"]: c for c in previous.get("criteria", [])}
    if previous.get("engine") != current.get("engine"):
        old = {}
    if previous.get("rule_signature") != current.get("rule_signature"):
        old = {}
    improved, remaining = [], []
    rank = {"missing": 0, "partial": 1, "met": 2}
    for criterion in current.get("criteria", []):
        earlier = old.get(criterion["id"])
        if earlier and earlier.get("reference") == criterion.get("reference"):
            before, after = earlier["status"], criterion["status"]
            if (earlier.get("assessment_reason") != "not_confirmed"
                    and before in rank and after in rank and rank[after] > rank[before]):
                improved.append(criterion["label"])
        if criterion["status"] in {"partial", "missing", "error", "unverified"}:
            terms = [item for item in current.get("missing_required_keywords", [])
                     if item.get("criterion_id") == criterion["id"]]
            if criterion["status"] == "missing" and terms:
                remaining.extend(item["reason"] for item in terms)
            elif criterion.get("assessment_reason") == "direction_error":
                remaining.append(_learning_text(language, "remaining_reversed", reference=criterion["reference"]))
            elif criterion.get("assessment_reason") == "not_confirmed":
                remaining.append(_learning_text(language, "remaining_unverified", reference=criterion["reference"]))
            else:
                remaining.append(_learning_text(language, "remaining_relation", reference=criterion["reference"]))
    return {"improved": improved, "remaining": list(dict.fromkeys(remaining))}


def recurring_misconceptions(attempts):
    """이 문항의 서로 다른 제출마다 한 번씩 집계하며 다른 사용자의 기록은 합치지 않는다."""
    groups = {}
    for attempt in attempts:
        seen = set()
        for misconception in (attempt.feedback_json or {}).get("learning", {}).get("misconceptions", []):
            key = misconception.get("key")
            if not key or key in seen:
                continue
            seen.add(key)
            learning = (attempt.feedback_json or {}).get("learning", {})
            criterion = next((c for c in learning.get("criteria", []) if c["id"] == misconception.get("criterion_id")), {})
            language = learning.get("language", LANG_KO)
            label = criterion.get("label", _learning_text(language, "default_label"))
            error_label = _error_labels(language).get(misconception.get("type"), _learning_text(language, "default_error"))
            group = groups.setdefault(key, {"message": f"{label}: {error_label}", "count": 0, "key": key})
            group["count"] += 1
    return [item for item in groups.values() if item["count"] >= 2]
