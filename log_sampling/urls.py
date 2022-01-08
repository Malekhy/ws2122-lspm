from django.conf.urls import url
from django.urls import path
from . import views

app_name = 'log_sampling'


urlpatterns = [
    path('', views.gogo, name='log_sampling'),

]


