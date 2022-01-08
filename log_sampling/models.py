from django.db import models
from django.utils.translation import gettext as _
from django.contrib.auth.models import User

# Create your models here.

class Characteristics(models.Model):
    CaseId = models.IntegerField(("Case ID"))
    ActivityName = models.CharField(_("Activity"),max_length=255)