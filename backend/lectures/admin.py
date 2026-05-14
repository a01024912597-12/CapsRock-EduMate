from django.contrib import admin
from .models import Lecture, UserProfile, Quiz, QuizQuestion, QuizAnswer


class QuizQuestionInline(admin.TabularInline):
    model = QuizQuestion
    extra = 0
    fields = ("number", "question_text", "model_answer", "explanation")
    ordering = ("number",)


class QuizAnswerInline(admin.TabularInline):
    model = QuizAnswer
    extra = 0
    fields = (
        "user",
        "user_answer",
        "similarity_score",
        "predicted_label",
        "human_score",
        "human_label",
        "created_at",
    )
    readonly_fields = ("created_at",)
    ordering = ("created_at",)


@admin.register(Lecture)
class LectureAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "user",
        "whisper_model_name",
        "analysis_duration_seconds",
        "stt_duration_seconds",
        "summary_duration_seconds",
        "analyzed_at",
        "created_at",
    )
    list_filter = ("whisper_model_name", "created_at", "analyzed_at")
    search_fields = ("title", "youtube_url", "summary_text")
    readonly_fields = (
        "created_at",
        "analyzed_at",
        "analysis_duration_seconds",
        "stt_duration_seconds",
        "summary_duration_seconds",
        "whisper_model_name",
    )


@admin.register(Quiz)
class QuizAdmin(admin.ModelAdmin):
    list_display = ("lecture", "generation_number", "created_at")
    list_filter = ("generation_number", "created_at")
    search_fields = ("lecture__title", "quiz_text")
    ordering = ("lecture", "generation_number")
    inlines = [QuizQuestionInline]


@admin.register(QuizQuestion)
class QuizQuestionAdmin(admin.ModelAdmin):
    list_display = (
        "get_lecture_title",
        "get_generation_number",
        "number",
        "short_question",
        "created_at",
    )
    list_filter = ("quiz__generation_number", "quiz__lecture", "created_at")
    search_fields = ("quiz__lecture__title", "question_text", "model_answer")
    ordering = ("quiz__lecture__title", "quiz__generation_number", "number")
    inlines = [QuizAnswerInline]

    def get_lecture_title(self, obj):
        return obj.quiz.lecture.title

    get_lecture_title.short_description = "강의 제목"

    def get_generation_number(self, obj):
        return obj.quiz.generation_number

    get_generation_number.short_description = "차수"

    def short_question(self, obj):
        if len(obj.question_text) > 40:
            return obj.question_text[:40] + "..."
        return obj.question_text

    short_question.short_description = "문제 내용"


@admin.register(QuizAnswer)
class QuizAnswerAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "get_lecture_title",
        "get_generation_number",
        "get_question_number",
        "short_user_answer",
        "similarity_score",
        "predicted_label",
        "human_score",
        "human_label",
        "created_at",
    )
    list_filter = (
        "predicted_label",
        "human_label",
        "quiz_question__quiz__generation_number",
        "quiz_question__quiz__lecture",
        "created_at",
    )
    search_fields = (
        "user__username",
        "quiz_question__quiz__lecture__title",
        "quiz_question__question_text",
        "user_answer",
    )
    ordering = (
        "quiz_question__quiz__lecture__title",
        "quiz_question__quiz__generation_number",
        "quiz_question__number",
        "created_at",
    )

    def get_lecture_title(self, obj):
        return obj.quiz_question.quiz.lecture.title

    get_lecture_title.short_description = "강의 제목"

    def get_generation_number(self, obj):
        return obj.quiz_question.quiz.generation_number

    get_generation_number.short_description = "차수"

    def get_question_number(self, obj):
        return obj.quiz_question.number

    get_question_number.short_description = "문제 번호"

    def short_user_answer(self, obj):
        if len(obj.user_answer) > 40:
            return obj.user_answer[:40] + "..."
        return obj.user_answer

    short_user_answer.short_description = "사용자 답안"


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "name", "age", "phone", "certification")
    search_fields = ("user__username", "name", "phone", "certification")