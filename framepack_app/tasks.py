from celery import shared_task
from django.core.files.base import ContentFile
import os
import traceback
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from django.core.files.storage import default_storage  # Import default_storage

from framepack_app.models import VideoGenerationTask
from FramePack.demo_gradio_f1 import process  # Import the process function

def load_image_from_url(image_url):
    """Downloads an image from a URL and returns it as a NumPy array."""
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        image = Image.open(BytesIO(response.content))
        return np.array(image)
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error downloading image: {e}")
    except Exception as e:
        raise Exception(f"Error opening image: {e}")

@shared_task(bind=True)
def generate_video_task(self, task_id, input_image_path, params):
    task = VideoGenerationTask.objects.get(id=task_id)
    try:
        # Download image from URL
        input_image_np = load_image_from_url(input_image_path)

        # Call the process function directly
        output_filename = process(
            input_image=input_image_np,
            prompt=params['prompt'],
            total_second_length=params['total_second_length']
        )

        # Upload to storage
        with open(output_filename, 'rb') as f:
            content = ContentFile(f.read())
            task.video_key = f'results/{task.id}.mp4'
            default_storage.save(task.video_key, content)  # Save to MinIO

        task.status = 'completed'  # Update status after successful upload
        task.save()

        # Cleanup
        os.remove(output_filename)  # Remove local file

    except Exception as e:
        task.status = 'failed'
        task.save()
        traceback.print_exc()
        raise self.retry(exc=e, countdown=5, max_retries=1)
