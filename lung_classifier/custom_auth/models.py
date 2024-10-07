from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from .enums import GenderEnum , RoleEnum

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
    
    def create_superuser(self, email , password, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(email, password, **extra_fields)


class User(AbstractUser):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    date_of_birth = models.DateField(null=True , blank=True)
    gender = models.CharField(max_length=10, choices=GenderEnum.choices(), null=True, blank=True)
    role = models.CharField(max_length=10, choices=RoleEnum.choices())
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name' , 'last_name']

    objects = UserManager()

    def __str__(self):
        return self.email
    
    groups = models.ManyToManyField(
        'auth.Group',
        related_name='coreuser_groups',  # Unique related name for groups
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions '
                  'granted to each of their groups.',
        verbose_name='groups',
    )

    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='coreuser_permissions',  # Unique related name for permissions
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions',
    )