from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('sampling', views.sampling, name='sampling')
]