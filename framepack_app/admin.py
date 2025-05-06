from django.contrib import admin
from .models import VideoGenerationTask

@admin.register(VideoGenerationTask)
class VideoGenerationAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'created_at')
    search_fields = ('id', 'status')