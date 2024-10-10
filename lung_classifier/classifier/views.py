import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from django.conf import settings
from keras.models import load_model
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser ,FormParser
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .serializers import ImageClassifierSerializer
from classifier.models import Image
from rest_framework.generics import ListAPIView , DestroyAPIView , RetrieveAPIView
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns

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
            generate_graph = self._generate_graphs(prediction)
            graph_urls = {
                "confusion_matrix": os.path.join(settings.MEDIA_URL, f"plots/{self.request.user.username}/confusion_matrix.png"),
                "roc_curve": os.path.join(settings.MEDIA_URL, f"plots/{self.request.user.username}/roc_curve.png"),
                "precision_recall_curve": os.path.join(settings.MEDIA_URL, f"plots/{self.request.user.username}/precision_recall_curve.png"),
            }
            
            serializer = ImageClassifierSerializer(image_instance)              

            return Response({
                "message": prediction["message"],
                "image_data": serializer.data,
                "graphs": graph_urls
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    def _generate_graphs(self, prediction):
        """Generates the required graphs after making a prediction."""
        # Create a unique directory for each user to store their plots
        user_folder = os.path.join(settings.MEDIA_ROOT, f'plots/{self.request.user.username}')
        os.makedirs(user_folder, exist_ok=True)
        
        predicted_class = prediction['prediction']
        
        y_true = [0] 
        y_pred = [0]  

        # Generate Confusion Matrix and save it to the user's folder
        self._plot_confusion_matrix(y_true, y_pred, classes=['lung_aca', 'lung_n', 'lung_scc'], user_folder=user_folder)
        self._plot_roc_curve(y_true, y_pred, user_folder=user_folder)

        # Generate Precision-Recall Curve
        self._plot_precision_recall_curve(y_true, y_pred, user_folder=user_folder)

    def _plot_confusion_matrix(self, y_true, y_pred, classes, user_folder):
        """Plots and saves the confusion matrix in the user folder."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        
        # Save confusion matrix to user's folder
        plot_path = os.path.join(user_folder, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        
    def _plot_confusion_matrix(self, y_true, y_pred, classes, user_folder):
        """Plots and saves the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        
        # Save confusion matrix
        plot_path = os.path.join(user_folder, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()

    def _plot_roc_curve(self, y_true, y_pred, user_folder):
        """Plots and saves the ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save ROC curve
        plot_path = os.path.join(user_folder, 'roc_curve.png')
        plt.savefig(plot_path)
        plt.close()

    def _plot_precision_recall_curve(self, y_true, y_pred, user_folder):
        """Plots and saves the precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        plt.figure()
        plt.plot(recall, precision, lw=2, color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        # Save precision-recall curve
        plot_path = os.path.join(user_folder, 'precision_recall_curve.png')
        plt.savefig(plot_path)
        plt.close()
        
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
    
    
class DeleteClassifiedImageView(DestroyAPIView):
    permission_classes = [IsAuthenticated,]
    
    def get_queryset(self):
        return Image.objects.filter(id= self.kwargs['pk'])
    
    def delete(self, request, *args, **kwargs):
        response = super().delete(request, *args, **kwargs)
        return Response(
            {
                'message': f'Image has been deleted successfully',
                'data':response.data
            }
        )
        
class GetClassifiedImageDetail(RetrieveAPIView):
    permission_classes = [IsAuthenticated,]
    serializer_class = ImageClassifierSerializer
    
    def get_queryset(self):
        return Image.objects.filter(id= self.kwargs['pk'])
    
    def retrieve(self, request, *args, **kwargs):
        response =  super().retrieve(request, *args, **kwargs)
        return Response({
            'message': 'Image Details has been fetched successfully',
            'data' : response.data
        })