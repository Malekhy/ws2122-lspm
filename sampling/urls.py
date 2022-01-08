from django.urls import path
from . import views



urlpatterns = [
    path('', views.run, name='sampling'),
    path('AjaxCall', views.AjaxCall, name='AjaxCall'),
    path('AjaxDownload', views.AjaxDownload, name='AjaxDownload'),



]



