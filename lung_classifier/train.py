import django
from django.conf import settings
import os
from classifier.ml.preproces_data import preprocess_images
from classifier.ml.model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lung_classifier.settings')

def train_model():
    print("Running Script train.py")
    path = settings.DATA_DIR
    classes = ['lung_n', 'lung_aca', 'lung_scc', 'non-lung']

    X, Y = preprocess_images(path, classes)
    Y = pd.get_dummies(Y).values

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

    model = build_model()

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=8, batch_size=16)
    model.save('new_lung_classifier_model.h5')

response = train_model()
print(response)