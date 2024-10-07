from django.contrib import admin
from .models import Image

class ImageAdmin(admin.ModelAdmin):
    list_display = ('user', 'image', 'created_at', 'prediction')
    list_filter = ('created_at', 'user')
    search_fields = ('user__email', 'prediction')

admin.site.register(Image, ImageAdmin)