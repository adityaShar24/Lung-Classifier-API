from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import (
    CreateAPIView,
    GenericAPIView,
)
from .serializers import (
    UserRegistrationSerializer,
    UserLoginSerializer,
    )

class UserRegisterView(CreateAPIView):
    serializer_class = UserRegistrationSerializer
    
    def create(self , request , *args , **kwargs):
        serializer = self.get_serializer(data= request.data)
        serializer.is_valid(raise_exception = True)
        user = serializer.save()
        return Response(
            {'message': 'User has been registered successfully.' , 'user': serializer.data},
            status=status.HTTP_201_CREATED
        )

class UserLoginView(GenericAPIView):
    serializer_class = UserLoginSerializer
    
    def post(self , request, *args , **kwargs):
        serializer = self.get_serializer(data = request.data)
        serializer.is_valid(rasie_exception = True)
        user = serializer.validated_data['user']
        
        return Response(
            {
                "message": "User has been logged in successfully.",
                "email": user.email,
                "role": user.role,
            },
            status=status.HTTP_200_OK
        )