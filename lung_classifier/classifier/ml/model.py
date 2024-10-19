# classifier/ml/model.py
from keras.applications import InceptionV3
from keras import layers
from keras.models import Model

IMG_SIZE = 128

def build_model():
    pre_trained_model = InceptionV3(input_shape=(IMG_SIZE, IMG_SIZE, 3), 
                                    weights='imagenet', 
                                    include_top=False)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    x = layers.Flatten()(last_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(4, activation='softmax')(x)

    model = Model(pre_trained_model.input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
