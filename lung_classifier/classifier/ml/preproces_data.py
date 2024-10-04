# classifier/ml/preprocess.py
import os
import cv2
import numpy as np

def preprocess_images(path, classes):
    images = []
    labels = []

    for label in classes:
        class_dir = os.path.join(path, label)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  
            images.append(img)
            labels.append(label)

    X = np.array(images) / 255.0 
    Y = np.array(labels)
    
    return X, Y
