# Generated by Django 4.1.3 on 2022-12-02 09:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0003_prediccion_created_at_prediccion_pred_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="prediccion", name="temp", field=models.FloatField(default=1),
        ),
    ]