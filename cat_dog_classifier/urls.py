from django.contrib import admin
from django.urls import path
from classifier import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('' , views.upload, name="upload"),
]
