from django.urls import path
from .views import (
    home,
    login_page,
    logout_page,
    signup_page,
    upload_page,
    summary_page,
    analyze_video_api,
    history_page,
    quiz_page,
    quiz_answer_page,
    feedback_page,
    test_api,
)

urlpatterns = [
    path('', home, name='home'),
    path('login/', login_page, name='login'),
    path('logout/', logout_page, name='logout'),
    path('signup/', signup_page, name='signup'),
    path('upload/', upload_page, name='upload'),
    path('summary/', summary_page, name='summary'),
    path('analyze-api/', analyze_video_api, name='analyze_api'),
    path('history/', history_page, name='history'),
    path('quiz/', quiz_page, name='quiz'),
    path('quiz/answer/', quiz_answer_page, name='quiz_answer'),
    path('feedback/', feedback_page, name='feedback'),
    path('api/test/', test_api, name='test_api'),
]