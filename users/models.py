from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.

class User(AbstractUser):
    rol = models.CharField(max_length=50, null=True)

class Prediccion(models.Model):
    pred = models.FloatField(default=1)
    temp = models.FloatField(default=1)
    conduc = models.IntegerField(default=1)
    difer = models.IntegerField(default=1)
    flujoA = models.IntegerField(default=1)
    user = models.TextField(default='none')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    def __str__(self):
        texto = "(temp: {0}) (conduc: {1}) (difer: {2}) (flujoA: {3})"
        return texto.format(self.temp, self.conduc, self.difer, self.flujoA)