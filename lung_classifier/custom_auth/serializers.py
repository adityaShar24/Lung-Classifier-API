from django.contrib.auth import authenticate
from rest_framework.serializers import (
    ModelSerializer,
    CharField,
    EmailField,
    Serializer,
    ValidationError
)

from .models import User

class UserRegistrationSerializer(ModelSerializer):
    password = CharField(write_only=True)

    class Meta:
        model = User
        fields = ['email', 'username', 'password', 'role']  
        extra_kwargs = {'password': {'write_only': True}} 
        
    def create(self, validated_data):
        user = User(
            email=validated_data['email'],
            username=validated_data['username'],
            role=validated_data['role'] 
        )
        user.set_password(validated_data['password']) 
        user.save()
        return user
    
class UserLoginSerializer(Serializer):
    email = EmailField(required=True)
    password = CharField(required=True, write_only=True)

    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')

        user = authenticate(email=email, password=password)
        if user is None:
            raise ValidationError("Invalid email or password.")
        
        attrs['user'] = user
        return attrs

    def to_representation(self, instance):
        """Customize the representation of the validated user data."""
        
        return {
            'email': instance['user'].email,
            'role': instance['user'].role,
        }