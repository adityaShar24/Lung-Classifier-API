from rest_framework.serializers import (
    ModelSerializer , 
    ValidationError ,
)
from .models import Image
from PIL import Image as PILImage 

class ImageClassifierSerializer(ModelSerializer):
    class Meta:
        model = Image
        fields = '__all__'
        read_only_fields = ['id', 'user', 'prediction']


