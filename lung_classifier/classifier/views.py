from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from keras.models import load_model
import numpy as np
import cv2
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status

ML_MODEL_PATH = settings.model_path

model = load_model(ML_MODEL_PATH)

class LungCancerClassifierView(APIView):
    parser_classes = [MultiPartParser]  # Handles multipart form data (for file uploads)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('image', None)

        if not file_obj:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            img_array = self._read_image(file_obj)
            prediction = self._make_prediction(img_array)
            
            # Return the result as JSON response
            return Response({"prediction": prediction}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _read_image(self, file_obj):
        """Converts the uploaded file to an OpenCV image and preprocesses it."""
        # Convert the uploaded image to a numpy array
        image_data = np.fromstring(file_obj.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        # Resize and normalize the image for the model
        img_size = 128  # Resize to 128x128 as per the model's requirement
        img = cv2.resize(img, (img_size, img_size))  # Resize to 128x128
        img = img.astype('float32') / 255.0  # Normalize the image to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        return img

    def _make_prediction(self, img_array):
        """Runs the model prediction and returns the class label."""
        classes = ['lung_n', 'lung_aca', 'lung_scc']  # Define your classes
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]
        return predicted_class