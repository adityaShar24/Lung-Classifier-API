from django.urls import path
from .views import (
    LungCancerClassifierView,
    GetAllClassifiedImages,
    )

urlpatterns = [ 
    path('classify-image/', LungCancerClassifierView.as_view(), name='classify-image'),
    path('get-classified-images/', GetAllClassifiedImages.as_view(), name='classified-images'),
]