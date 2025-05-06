from django.urls import path
from .views import GenerateVideoView, TaskStatusView

urlpatterns = [
    path('api/generate/', GenerateVideoView.as_view(), name='generate'),
    path('api/task/<uuid:task_id>/', TaskStatusView.as_view(), name='task-status'),
]