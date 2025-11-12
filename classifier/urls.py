from django.urls import path
from classifier import views

urlpatterns = [
    path('' , views.upload, name="upload"),
]
