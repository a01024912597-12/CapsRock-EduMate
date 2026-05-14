import os
import django


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()


from lectures.models import QuizAnswer


def normalize_label(label):
    """라벨 표현을 비교 가능한 공통 라벨로 변환한다."""
    if not label:
        return None

    label = str(label).strip()

    if label in ["정답", "정답 가능성 높음"]:
        return "정답"

    if label in ["부분정답", "검토 필요"]:
        return "부분정답"

    if label in ["오답", "오답 가능성 높음"]:
        return "오답"

    return None


def label_from_human_score(score):
    """human_label이 없을 때 human_score로 사람 판단 라벨을 추정한다."""
    if score is None:
        return None

    try:
        score = float(score)
    except (TypeError, ValueError):
        return None

    if score >= 75:
        return "정답"

    if score >= 45:
        return "부분정답"

    return "오답"


def get_human_label(answer):
    """QuizAnswer에서 사람 검토 라벨을 가져온다.

    우선순위:
    1. human_label
    2. human_score 기반 추정
    """
    label = normalize_label(answer.human_label)

    if label:
        return label

    return label_from_human_score(answer.human_score)


def predict_label_by_threshold(similarity_score, correct_threshold, partial_threshold):
    """threshold 기준으로 시스템 예측 라벨을 만든다."""
    if similarity_score >= correct_threshold:
        return "정답"

    if similarity_score >= partial_threshold:
        return "부분정답"

    return "오답"


def collect_reviewed_answers():
    """사람 검토가 완료된 답안만 수집한다."""
    answers = (
        QuizAnswer.objects
        .filter(similarity_score__isnull=False)
        .select_related(
            "user",
            "quiz_question",
            "quiz_question__quiz",
            "quiz_question__quiz__lecture",
        )
        .order_by(
            "quiz_question__quiz__lecture__title",
            "quiz_question__quiz__generation_number",
            "quiz_question__number",
            "created_at",
        )
    )

    reviewed = []

    for answer in answers:
        human_label = get_human_label(answer)

        if not human_label:
            continue

        reviewed.append({
            "id": answer.id,
            "lecture_title": answer.quiz_question.quiz.lecture.title,
            "generation_number": answer.quiz_question.quiz.generation_number,
            "question_number": answer.quiz_question.number,
            "user_answer": answer.user_answer,
            "similarity_score": float(answer.similarity_score or 0),
            "current_predicted_label": normalize_label(answer.predicted_label),
            "human_label": human_label,
            "human_score": answer.human_score,
        })

    return reviewed


def evaluate_threshold(reviewed_answers, correct_threshold, partial_threshold):
    """특정 threshold 조합의 사람 판단 일치율을 계산한다."""
    total = len(reviewed_answers)

    if total == 0:
        return {
            "correct_threshold": correct_threshold,
            "partial_threshold": partial_threshold,
            "total": 0,
            "match_count": 0,
            "accuracy": 0,
        }

    match_count = 0

    for item in reviewed_answers:
        predicted = predict_label_by_threshold(
            item["similarity_score"],
            correct_threshold,
            partial_threshold,
        )

        if predicted == item["human_label"]:
            match_count += 1

    accuracy = round((match_count / total) * 100, 2)

    return {
        "correct_threshold": correct_threshold,
        "partial_threshold": partial_threshold,
        "total": total,
        "match_count": match_count,
        "accuracy": accuracy,
    }


def find_best_thresholds(reviewed_answers):
    """여러 threshold 조합을 비교해서 일치율이 높은 순서대로 반환한다."""
    results = []

    # 정답 기준: 0.50 ~ 0.95
    # 부분정답 기준: 0.20 ~ 정답 기준보다 낮은 값
    for correct_i in range(50, 96, 5):
        correct_threshold = correct_i / 100

        for partial_i in range(20, correct_i, 5):
            partial_threshold = partial_i / 100

            result = evaluate_threshold(
                reviewed_answers,
                correct_threshold,
                partial_threshold,
            )
            results.append(result)

    results.sort(
        key=lambda x: (
            x["accuracy"],
            x["match_count"],
            -abs(x["correct_threshold"] - 0.75),
            -abs(x["partial_threshold"] - 0.45),
        ),
        reverse=True,
    )

    return results


def print_reviewed_answers(reviewed_answers):
    print("\n[사람 검토 데이터 목록]")
    print("-" * 80)

    for item in reviewed_answers:
        print(
            f"ID={item['id']} | "
            f"{item['lecture_title']} | "
            f"{item['generation_number']}차 {item['question_number']}번 | "
            f"유사도={item['similarity_score']:.4f} | "
            f"현재 시스템={item['current_predicted_label']} | "
            f"사람={item['human_label']} | "
            f"human_score={item['human_score']}"
        )

    print("-" * 80)


def print_threshold_result(title, result):
    print(f"\n[{title}]")
    print(f"정답 기준: {result['correct_threshold']:.2f}")
    print(f"부분정답 기준: {result['partial_threshold']:.2f}")
    print(f"검토 답안 수: {result['total']}개")
    print(f"일치 개수: {result['match_count']}개")
    print(f"일치율: {result['accuracy']}%")


def print_mismatch_examples(reviewed_answers, correct_threshold, partial_threshold):
    print("\n[불일치 사례]")
    print("-" * 80)

    mismatch_count = 0

    for item in reviewed_answers:
        predicted = predict_label_by_threshold(
            item["similarity_score"],
            correct_threshold,
            partial_threshold,
        )

        if predicted == item["human_label"]:
            continue

        mismatch_count += 1

        print(
            f"ID={item['id']} | "
            f"{item['lecture_title']} | "
            f"{item['generation_number']}차 {item['question_number']}번"
        )
        print(f"유사도: {item['similarity_score']:.4f}")
        print(f"시스템 예측: {predicted}")
        print(f"사람 판단: {item['human_label']}")
        print(f"사용자 답안: {item['user_answer'][:120]}")
        print("-" * 80)

    if mismatch_count == 0:
        print("불일치 사례가 없습니다.")
        print("-" * 80)


def main():
    reviewed_answers = collect_reviewed_answers()

    if not reviewed_answers:
        print("\n사람 검토 데이터가 없습니다.")
        print("관리자 페이지에서 QuizAnswer의 human_label 또는 human_score를 먼저 입력하세요.")
        print("추천 human_label 값: 정답 / 부분정답 / 오답")
        return

    print_reviewed_answers(reviewed_answers)

    current_result = evaluate_threshold(
        reviewed_answers,
        correct_threshold=0.75,
        partial_threshold=0.45,
    )
    print_threshold_result("현재 기준 평가", current_result)

    results = find_best_thresholds(reviewed_answers)

    print("\n[추천 threshold TOP 10]")
    print("-" * 80)

    for idx, result in enumerate(results[:10], start=1):
        print(
            f"{idx}. "
            f"정답 기준={result['correct_threshold']:.2f}, "
            f"부분정답 기준={result['partial_threshold']:.2f}, "
            f"일치={result['match_count']}/{result['total']}, "
            f"일치율={result['accuracy']}%"
        )

    print("-" * 80)

    best_result = results[0]
    print_threshold_result("최적 기준 평가", best_result)

    print_mismatch_examples(
        reviewed_answers,
        correct_threshold=best_result["correct_threshold"],
        partial_threshold=best_result["partial_threshold"],
    )

    print("\n[views.py 반영 예시]")
    print("아래 기준을 predict_answer_label()에 반영할 수 있습니다.")
    print()
    print("def predict_answer_label(similarity_score):")
    print(f"    if similarity_score >= {best_result['correct_threshold']:.2f}:")
    print('        return "정답 가능성 높음"')
    print()
    print(f"    if similarity_score >= {best_result['partial_threshold']:.2f}:")
    print('        return "검토 필요"')
    print()
    print('    return "오답 가능성 높음"')


if __name__ == "__main__":
    main()