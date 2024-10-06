from django.contrib.auth.models import BaseUserManager
from django.db import models


class UserManager(BaseUserManager):
    use_in_migrations = True
    
    def _create_user(self , email , password , **extra_fields):
        if not email:
            raise ValueError("The Email Field must be set")
        if not password:
            raise ValueError("The Password Field must be set")
        email = self.normalize_email(email)
        username = extra_fields.pop("username") if "username" in extra_fields else email
        user = self.model(username= username, email = email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user , True
    
    def create_superuser(self, phone, password, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(phone, password, **extra_fields)
    