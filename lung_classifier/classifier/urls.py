from django.urls import path
from .views import LungCancerClassifierView

urlpatterns = [ 
    path('classify-image/', LungCancerClassifierView.as_view(), name='classify-image'),
]