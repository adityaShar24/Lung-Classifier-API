from django.urls import path
from .views import (
    UserRegisterView,
    UserLoginView
)


urlpatterns = [
    path('register/', UserRegisterView.as_view(), name='register-user'),
    path('login/', UserLoginView.as_view(), name='login-user'),
]