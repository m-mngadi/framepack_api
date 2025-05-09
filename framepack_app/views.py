from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from framepack_app.models import VideoGenerationTask
from .serializers import GenerateVideoSerializer, TaskStatusSerializer
from .tasks import generate_video_task

class GenerateVideoView(APIView):
    def post(self, request):
        serializer = GenerateVideoSerializer(data=request.data)
        if serializer.is_valid():
            # Get image URL from request
            input_image_url = serializer.validated_data['input_image']  # Changed to URL

            # Create task record
            task = VideoGenerationTask.objects.create(status='in_progress')

            # Trigger Celery task
            generate_video_task.delay(
                str(task.id),
                input_image_url,  # Pass the URL
                serializer.validated_data
            )

            return Response({'task_id': str(task.id)}, status=status.HTTP_202_ACCEPTED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TaskStatusView(APIView):
    def get(self, request, task_id):
        try:
            task = VideoGenerationTask.objects.get(id=task_id)
            serializer = TaskStatusSerializer(task)
            return Response(serializer.data)
        except VideoGenerationTask.DoesNotExist:
            return Response({'error': 'Task not found'}, status=status.HTTP_404_NOT_FOUND)
