from django.db import models
from django.contrib.auth.models import User


class UserPreferences(models.Model):
    user_id   = models.OneToOneField(User, primary_key=True, on_delete=models.CASCADE)


    vegetables = models.CharField(max_length=4, default='true')
    gluten     = models.CharField(max_length=4, default='true')
    dairy      = models.CharField(max_length=4, default='true')
    pork       = models.CharField(max_length=4, default='true')
    oven       = models.CharField(max_length=4, default='true')
    microwave  = models.CharField(max_length=4, default='true')
    blender    = models.CharField(max_length=4, default='true')


    # vegetables = models.CharField(max_length=4, choices=[('true', 'true'), ('', '')], default='true')
    # gluten     = models.CharField(max_length=4, choices=[('true', 'true'), ('', '')], default='true')
    # dairy      = models.CharField(max_length=4, choices=[('true', 'true'), ('', '')], default='true')
    # pork       = models.CharField(max_length=4, choices=[('true', 'true'), ('', '')], default='true')
    # oven       = models.CharField(max_length=4, choices=[('true', 'true'), ('', '')], default='true')
    # microwave  = models.CharField(max_length=4, choices=[('true', 'true'), ('', '')], default='true')
    # blender    = models.CharField(max_length=4, choices=[('true', 'true'), ('', '')], default='true')






