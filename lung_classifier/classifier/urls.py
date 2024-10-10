from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import (
    LungCancerClassifierView,
    GetAllClassifiedImages,
    DeleteClassifiedImageView,
    GetClassifiedImageDetail,
    )

urlpatterns = [ 
    path('classify-image/', LungCancerClassifierView.as_view(), name='classify-image'),
    path('get-classified-images/', GetAllClassifiedImages.as_view(), name='classified-images'),
    path('delete-classified-image/<int:pk>/', DeleteClassifiedImageView.as_view(), name='delete-classified-image'),
    path('detail-classified-image/<int:pk>/', GetClassifiedImageDetail.as_view(), name='detail-classified-image'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)