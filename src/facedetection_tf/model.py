import keras as K
import tensorflow as tf
from keras._tf_keras.keras.layers import GlobalMaxPooling2D, Dense, Input
from keras._tf_keras.keras import Model
from keras._tf_keras.keras.applications.vgg16 import VGG16

from src.facedetection_tf.logger import FD_Logger
from src.facedetection_tf.exception import FD_Exception


from dataclasses import dataclass, field
from typing import Tuple, Annotated
import os, sys


logger = FD_Logger().logger()


@dataclass
class FaceDetectionModel:
    input_shape:Tuple[int]
    vgg:Model=field(default=VGG16(include_top=False))

    def build_model(self):
        try:
            
            input_layer = Input(shape=self.input_shape)
            vgg_model = self.vgg(input_layer)

            logger.info("VGG16 model initialized with the input layer...")

            f1 = GlobalMaxPooling2D()(vgg_model)

            ##          classifier
            class1 = Dense(2048, activation='relu')(f1)
            class2 = Dense(1, activation='sigmoid')(class1)

            ##          regressor
            regress1=Dense(2048, activation='relu')(f1)
            regress2=Dense(4, activation='sigmoid')(regress1)

            facetracker = Model(inputs=input_layer, outputs=[class2, regress2])

            logger.info("FaceDetectionModel model intialized...")

            return facetracker
        except Exception as e:
            logger.info(FD_Exception(e,sys))
            raise FD_Exception(e,sys)

if __name__=="__main__":
    __all__=["FaceDetectionModel","get_optimizer","get_localization_loss","get_categorical_loss"]