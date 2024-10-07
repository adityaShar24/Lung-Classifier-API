from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from keras.models import load_model
import numpy as np
import cv2
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView 
from rest_framework.parsers import MultiPartParser ,FormParser
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .serializers import ImageClassifierSerializer
from classifier.models import Image

ML_MODEL_PATH = settings.MODEL_PATH

model = load_model(ML_MODEL_PATH)

class LungCancerClassifierView(APIView):
    permission_classes = [IsAuthenticated,]  
    parser_classes = [MultiPartParser, FormParser] 
    
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('image', None)
        user = self.request.user
        
        print("File", file_obj)
        print("user", user)  
        
        if not file_obj:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            img_array = self._read_image(file_obj)
            prediction = self._make_prediction(img_array)
            
            image_instance = Image(user=user, image=file_obj, prediction=prediction["prediction"])
            image_instance.save()
            
            serializer = ImageClassifierSerializer(image_instance)              

            return Response({
                "message": prediction["message"],
                "image_data": serializer.data
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _read_image(self, file_obj):
        """Converts the uploaded file to an OpenCV image and preprocesses it."""
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
        classes = ['lung_aca', 'lung_n', 'lung_scc']  # Define your classes
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        print("Prediction",prediction, np.argmax(prediction))
        predicted_class = classes[predicted_index]
        accuracy = round(prediction[0][predicted_index] * 100, 2)
        
        if predicted_class == classes[1]:  # Check if the predicted class is 'lung_n'
            return {
                "prediction": predicted_class, 
                "message": f"The image shows normal lung tissue with accuracy of {accuracy}% "
                }
        elif predicted_class == classes[0]:
            return {"prediction": predicted_class, "message": f"The image shows signs of lung adenocarcinoma (lung_aca) with accuracy of {accuracy}% . Further examination is recommended."}
        elif predicted_class == classes[2]:
            return {"prediction": predicted_class, "message": f"The image shows signs of lung squamous cell carcinoma (lung_scc) with accuracy of {accuracy}% . Further examination is recommended."}
        else:
            return {"prediction": predicted_class, "message": "Unrecognized result. Please consult a specialist."}
        
        
class GetAllClassifiedImages(ListAPIView):
    permission_classes = [IsAuthenticated,]
    queryset = Image.objects.all()
    serializer_class = ImageClassifierSerializer
    
    def get_queryset(self):
        return super().get_queryset().filter(user=self.request.user)
    
    def list(self , request , *args, **kwargs):
        
        response = super().list(request, *args, **kwargs)
        return Response({
            'message':'Classified Images fetched successfully',
            'data': response.data
        })
    
    