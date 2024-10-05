from django.urls import path
from .views import LungCancerClassifierView

url_patterns = [
    path('classify-image', LungCancerClassifierView.as_view() , name='classify-image'),
]