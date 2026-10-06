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
