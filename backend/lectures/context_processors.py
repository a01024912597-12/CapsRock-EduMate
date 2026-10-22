from .classroom_views import user_classrooms
from .permissions import is_instructor


def edumate_ui(request):
    if request.user.is_authenticated:
        display_name = ""
        try:
            display_name = request.user.profile.name.strip()
        except Exception:
            pass
        if not display_name:
            display_name = request.user.get_full_name().strip() or request.user.username
        initial = display_name[0] if display_name else "?"
    else:
        display_name = ""
        initial = ""

    return {
        "user_display_name": display_name,
        "user_display_initial": initial,
    }


def classroom_sidebar(request):
    if not request.user.is_authenticated:
        return {}

    match = getattr(request, "resolver_match", None)
    current_id = None
    if match and match.url_name == "classroom_detail":
        current_id = match.kwargs.get("pk")

    return {
        "sidebar_is_instructor": is_instructor(request.user),
        "sidebar_classrooms": list(user_classrooms(request.user)),
        "sidebar_current_classroom_id": current_id,
    }
