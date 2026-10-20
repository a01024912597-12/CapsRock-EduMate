from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("lectures", "0014_remove_quizanswer_updated_at_quizquestion_updated_at")]
    operations = [
        migrations.AddField(
            model_name="quizquestion", name="feedback_rubric",
            field=models.JSONField(default=list, blank=True),
        ),
        migrations.CreateModel(
            name="QuizAnswerAttempt",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("user_answer", models.TextField(blank=True, default="")),
                ("score", models.FloatField(blank=True, null=True)),
                ("predicted_label", models.CharField(blank=True, max_length=50)),
                ("feedback_json", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("answer", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="attempts", to="lectures.quizanswer")),
            ],
            options={"ordering": ["created_at", "pk"]},
        ),
    ]
