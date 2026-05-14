from django.db import models
from django.contrib.auth.models import User


class Lecture(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='lectures')
    title = models.CharField(max_length=200)
    youtube_url = models.URLField()
    video_id = models.CharField(max_length=50, blank=True)
    thumbnail_url = models.URLField(blank=True)
    summary_text = models.TextField(blank=True)
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

    # GPT가 생성한 원본 퀴즈 텍스트 전체 저장
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
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.quiz} - {self.number}번 문제"

class QuizAnswer(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="quiz_answers")
    quiz_question = models.ForeignKey(QuizQuestion, on_delete=models.CASCADE, related_name="answers")

    user_answer = models.TextField(default="", blank=True)

    similarity_score = models.FloatField(null=True, blank=True)
    predicted_label = models.CharField(max_length=30, blank=True)

    human_score = models.FloatField(null=True, blank=True)
    human_label = models.CharField(max_length=30, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.quiz_question}"


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    name = models.CharField(max_length=100, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    certification = models.CharField(max_length=200, blank=True)
    reason = models.TextField(blank=True)
    interest = models.TextField(blank=True)

    def __str__(self):
        return f"{self.user.username}의 프로필"