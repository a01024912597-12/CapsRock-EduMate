from .models import ClassroomMembership, UserProfile


def is_instructor(user, profile=None):
    """강의자 권한 여부. 운영자(is_staff)와는 별개의 권한이다."""
    if not user.is_authenticated or not user.is_active:
        return False
    if profile is None:
        profile = UserProfile.objects.filter(user=user).only("is_instructor").first()
    return bool(profile and profile.is_instructor)


def is_classroom_member(user, classroom):
    return ClassroomMembership.objects.filter(classroom=classroom, user=user).exists()


def can_view_lecture(user, lecture):
    """강의 화면(영상)을 열 수 있는지.

    - 본인 강의는 항상 가능
    - 강의실 게시 강의는 강의자·참여 학생·운영자
    - 학생 사본은 그 학생과 강의실 강의자·운영자
    """
    if not user.is_authenticated:
        return False
    if lecture.user_id == user.id:
        return True
    classroom = lecture.classroom
    if classroom is None:
        return False
    if classroom.owner_id == user.id or user.is_staff:
        return True
    return lecture.is_classroom_post and is_classroom_member(user, classroom)


def can_view_summary(user, lecture):
    """요약 결과·PDF를 볼 수 있는지. 요약은 각자 하므로 게시 강의의 요약은 공유하지 않는다."""
    if not user.is_authenticated:
        return False
    if lecture.user_id == user.id or user.is_staff:
        return True
    classroom = lecture.classroom
    return bool(classroom and lecture.source_lecture_id and classroom.owner_id == user.id)
