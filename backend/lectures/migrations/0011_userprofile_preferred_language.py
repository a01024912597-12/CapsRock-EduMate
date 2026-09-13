from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("lectures", "0010_studycalendarmemo"),
    ]

    operations = [
        migrations.AddField(
            model_name="userprofile",
            name="preferred_language",
            field=models.CharField(
                choices=[
                    ("ko", "한국어"),
                    ("en", "English"),
                    ("zh-hans", "中文"),
                ],
                default="ko",
                max_length=10,
            ),
        ),
    ]
