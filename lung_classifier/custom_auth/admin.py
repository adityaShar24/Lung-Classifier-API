from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _

from .models import User

class UserAdmin(BaseUserAdmin):
    list_display = ('email', 'first_name', 'last_name', 'role', 'is_active', 'is_staff', 'is_superuser')
    
    # Fields to filter the list by in the adminK
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'role')
    
    # Ordering of the displayed fields in the admin
    ordering = ('email',)
    
    # Fields to display on the form page
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        (_('Personal Info'), {'fields': ('first_name', 'last_name', 'role')}),
        (_('Permissions'), {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        (_('Important dates'), {'fields': ('last_login', 'date_joined')}),
    )
    
    # Fields to include when creating a new user
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'first_name', 'last_name', 'role'),
        }),
    )

    # To search by email or name in admin
    search_fields = ('email', 'first_name', 'last_name')

# Register the custom user admin
admin.site.register(User, UserAdmin)
