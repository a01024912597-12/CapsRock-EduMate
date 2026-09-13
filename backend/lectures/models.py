from django.db import models
from django.contrib.auth.models import User


class Lecture(models.Model):
    SOURCE_YOUTUBE = "youtube"
    SOURCE_FILE = "file"
    SOURCE_CHOICES = [
        (SOURCE_YOUTUBE, "YouTube"),
        (SOURCE_FILE, "파일"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='lectures')
    title = models.CharField(max_length=200)
    source_type = models.CharField(
        max_length=20,
        choices=SOURCE_CHOICES,
        default=SOURCE_YOUTUBE,
    )
    youtube_url = models.URLField(blank=True)
    video_file = models.FileField(upload_to="lecture_videos/%Y/%m/", blank=True, null=True)
    video_id = models.CharField(max_length=50, blank=True)
    thumbnail_url = models.URLField(blank=True)
    summary_text = models.TextField(blank=True)

    summary_timeline = models.TextField(
        blank=True,
        help_text="키워드별 타임라인, 줄마다 '키워드 (분:초)' 형식",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    # 분석 시간 기록용 필드
    analysis_duration_seconds = models.FloatField(null=True, blank=True)
    stt_duration_seconds = models.FloatField(null=True, blank=True)
    summary_duration_seconds = models.FloatField(null=True, blank=True)
    whisper_model_name = models.CharField(max_length=50, blank=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.title


class Quiz(models.Model):
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE, related_name='quizzes')
    generation_number = models.PositiveIntegerField(default=1)

    # 기존 코드와 호환을 위해 유지
    question = models.TextField(default="", blank=True)
    answer = models.TextField(default="", blank=True)
    explanation = models.TextField(default="", blank=True)

    # AI가 생성한 원본 퀴즈 텍스트 전체 저장
    quiz_text = models.TextField(default="", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.lecture.title} - {self.generation_number}차 예상 문제"


class QuizQuestion(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='questions')
    number = models.PositiveIntegerField(default=1)
    question_text = models.TextField(default="", blank=True)
    model_answer = models.TextField(default="", blank=True)
    explanation = models.TextField(default="", blank=True)

    # 타임라인 기능 필드
    related_timeline = models.CharField(
        max_length=500,
        blank=True,
        help_text="요약문의 (분:초) 타임라인 인용",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.quiz} - {self.number}번 문제"


class QuizAnswer(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="quiz_answers")
    quiz_question = models.ForeignKey(QuizQuestion, on_delete=models.CASCADE, related_name="answers")

    user_answer = models.TextField(default="", blank=True)

    # 이름은 similarity_score지만 현재는 Reranker 기반 자동 채점 최종 점수 저장
    similarity_score = models.FloatField(null=True, blank=True)
    predicted_label = models.CharField(max_length=50, blank=True)

    # 사람 검토용 필드
    human_score = models.FloatField(null=True, blank=True)
    human_label = models.CharField(max_length=50, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.quiz_question}"


class UserProfile(models.Model):
    LANG_KO = "ko"
    LANG_EN = "en"
    LANG_ZH = "zh-hans"
    LANGUAGE_CHOICES = [
        (LANG_KO, "한국어"),
        (LANG_EN, "English"),
        (LANG_ZH, "中文"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    name = models.CharField(max_length=100, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    certification = models.CharField(max_length=200, blank=True)
    reason = models.TextField(blank=True)
    interest = models.TextField(blank=True)
    preferred_language = models.CharField(
        max_length=10,
        choices=LANGUAGE_CHOICES,
        default=LANG_KO,
    )

    def __str__(self):
        return f"{self.user.username}의 프로필"


class StudyCalendarMemo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="calendar_memos")
    date = models.DateField()
    memo = models.TextField(max_length=500, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user", "date"], name="unique_user_calendar_memo_date"),
        ]
        ordering = ["-date"]

    def __str__(self):
        return f"{self.user.username} - {self.date}"
