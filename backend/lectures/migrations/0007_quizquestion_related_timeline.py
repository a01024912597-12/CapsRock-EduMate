from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("lectures", "0006_quizanswer"),
    ]

    operations = [
        migrations.AddField(
            model_name="quizquestion",
            name="related_timeline",
            field=models.CharField(
                blank=True,
                help_text="요약문의 (분:초) 타임라인 인용",
                max_length=500,
            ),
        ),
    ]