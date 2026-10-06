from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("lectures", "0009_lecture_source_type_video_file"),
    ]

    operations = [
        migrations.CreateModel(
            name="StudyCalendarMemo",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("date", models.DateField()),
                ("memo", models.TextField(blank=True, max_length=500)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="calendar_memos",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-date"],
            },
        ),
        migrations.AddConstraint(
            model_name="studycalendarmemo",
            constraint=models.UniqueConstraint(
                fields=("user", "date"),
                name="unique_user_calendar_memo_date",
            ),
        ),
    ]
