import numpy as np
import base64
import io


import tensorflow as tf
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image


class Detection:
    def __init__(self):
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=(150,150,3),
                                                            include_top=False,
                                                            weights='imagenet')
        self.base_model.trainable = False
        #build the model
        self.model = Sequential()
        self.model.add(self.base_model)
        self.model.add(Layers.GlobalAveragePooling2D())
        self.model.add(Layers.Dense(128, activation='relu'))
        self.model.add(Layers.Dropout(0.2))
        self.model.add(Layers.Dense(1, activation = 'sigmoid'))
        self.model.load_weights('./model/weights/model_pneumonia_detection.h5')
        
    def process_image(self,path):
        img = tf.image.decode_jpeg(path, channels=3)
        img = tf.cast(img, tf.float32)
        img = img /255.
        img = tf.image.resize(img, (150,150))
        img = tf.expand_dims(img, 0)
        return img
    
    def read_img(self, path):
        img = image.load_img(path, target_size=(150,150))
        print(img)
        img = image.img_to_array(img, dtype= np.float32)
        #img = tf.cast(img, tf.float32)
        img = img /255.
        img = tf.expand_dims(img, 0)
        return img
    
    def predict(self, img):
        tensor = self.process_image(img)
        prediction = self.model.predict(tensor).flatten()[0]
        return prediction