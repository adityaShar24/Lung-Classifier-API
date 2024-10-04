from django.shortcuts import render
from django.http import JsonResponse
from keras.models import load_model
import numpy as np
import cv2

model = load_model('')