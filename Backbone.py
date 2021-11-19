import numpy as np
import tensorflow  as tf
from tensorflow.keras import layers 


class BackboneNetwork(tf.keras.Model):
    def __init__(self, batch_size, input_shape, network_type=None):
        super().__init__()
        
        self.batch_size = batch_size
        self.base_model_input_shape = input_shape
        self._network_type = network_type
        
        self.preprocess = layers.experimental.preprocessing.Rescaling(1./255, input_shape=(batch_size, *input_shape))
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(batch_size, *input_shape))
        self.pool1 = layers.MaxPooling2D(pool_size=2)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, activation="relu")
        self.pool2 = layers.MaxPooling2D(pool_size=2)
        self.conv3 = layers.Conv2D(filters=128, kernel_size=3, activation="relu")
        self.pool3 = layers.MaxPooling2D(pool_size=2)
        self.conv4 = layers.Conv2D(filters=256, kernel_size=3, activation="relu")
        self.pool4 = layers.MaxPooling2D(pool_size=2)
        self.conv5 = layers.Conv2D(filters=256, kernel_size=3, activation="relu")
        self.flatten = layers.Flatten()
    
    def call(self, state, preprocess=True):
        
        if preprocess:
            state = self.preprocess(state)
        
        x = self.conv1(state)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        output = self.flatten(x)
        
        return output