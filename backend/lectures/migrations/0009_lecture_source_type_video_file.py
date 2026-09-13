from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("lectures", "0008_lecture_summary_timeline_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="lecture",
            name="source_type",
            field=models.CharField(
                choices=[("youtube", "YouTube"), ("file", "파일")],
                default="youtube",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="lecture",
            name="video_file",
            field=models.FileField(
                blank=True,
                null=True,
                upload_to="lecture_videos/%Y/%m/",
            ),
        ),
        migrations.AlterField(
            model_name="lecture",
            name="youtube_url",
            field=models.URLField(blank=True),
        ),
    ]
