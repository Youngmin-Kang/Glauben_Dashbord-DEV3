# Generated by Django 4.1.2 on 2022-12-13 21:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0005_prediccion_user"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="rol",
            field=models.CharField(max_length=50, null=True),
        ),
    ]
