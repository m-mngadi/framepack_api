from rest_framework import serializers
from .models import VideoGenerationTask
from django.core.files.storage import default_storage

class GenerateVideoSerializer(serializers.Serializer):
    input_image = serializers.ImageField(required=True)
    prompt = serializers.CharField(required=True)
    n_prompt = serializers.CharField(default="")
    seed = serializers.IntegerField(default=31337)
    total_second_length = serializers.FloatField(default=5, min_value=1, max_value=120)
    latent_window_size = serializers.IntegerField(default=9, min_value=1, max_value=33)
    steps = serializers.IntegerField(default=25, min_value=1, max_value=100)
    cfg = serializers.FloatField(default=1.0, min_value=1.0, max_value=32.0)
    gs = serializers.FloatField(default=10.0, min_value=1.0, max_value=32.0)
    rs = serializers.FloatField(default=0.0, min_value=0.0, max_value=1.0)
    gpu_memory_preservation = serializers.FloatField(default=6, min_value=6, max_value=128)
    use_teacache = serializers.BooleanField(default=True)
    mp4_crf = serializers.IntegerField(default=16, min_value=0, max_value=100)

class TaskStatusSerializer(serializers.ModelSerializer):
    task_status = serializers.SerializerMethodField()
    video_url = serializers.SerializerMethodField()

    class Meta:
        model = VideoGenerationTask
        fields = ['task_id', 'task_status', 'video_url']
        read_only_fields = ['task_id', 'task_status', 'video_url']

    def get_task_status(self, obj):
        return obj.status.upper()

    def get_video_url(self, obj):
        if obj.video_key:
            try:
                return default_storage.url(obj.video_key)
            except:
                return None
        return None

    def to_representation(self, instance):
        data = super().to_representation(instance)
        if data['video_url'] is None:
            data.pop('video_url')
        return data

    def get_fields(self):
        fields = super().get_fields()
        fields['task_id'] = serializers.CharField(source='id')
        return fields