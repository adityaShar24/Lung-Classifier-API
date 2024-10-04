import cv2
import numpy as np
from glob import glob

IMG_SIZE = 128

def preprocess_image(path, classes):
    X = []
    Y = []
    
    for i , cat in enumerate(classes):
        images = glob(f'{path}/{cat}/*.jpeg')

        for image in images:
            img = cv2.imread(image)
            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            Y.append(i)

    return np.array(X), np.array(Y)