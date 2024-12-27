from keras._tf_keras.keras import Model
import tensorflow as tf
from src.facedetection_tf.model import FaceDetectionModel
from src.facedetection_tf.utils import get_optimizer, get_categorical_loss, get_localization_loss
from src.facedetection_tf.load_augmented_datasets import FD_Logger
from src.facedetection_tf.exception import FD_Exception

from dataclasses import dataclass, field

logger =FD_Logger().logger

class FaceTracker(tf.keras.Model):

    def __init__(self, facedetector_model:Model,**kwargs):
        super().__init__(**kwargs)
        self.model = facedetector_model
    
    def compile(self, optimizer:K.optimizers, cat_loss_func:K.losses.BinaryCrossentropy, local_loss_func:Any,**kwargs):
        super().compile(**kwargs)
        self.cat_loss = cat_loss_func
        self.optimizer = optimizer
        self.local_loss = local_loss_func
        logger.info("<<<< Compiler Initialized >>>>")

    def train_step(self,data,**kwargs):
        X, y = data

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            batch_cat_loss = self.cat_loss(y[0],classes)
            batch_local_loss = self.local_loss(tf.cast(y[1],tf.float32),coords)

            total_loss = batch_local_loss + (0.5 * batch_cat_loss)

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

        logger.info(f"<<<< Training Step: {'total_loss':total_loss, 'catagorical_loss':batch_cat_loss, 'localized_loss':batch_local_loss} >>>>")
        return {"total_loss":total_loss, "catagorical_loss":batch_cat_loss, "localized_loss":batch_local_loss}

    
    def test_step(self,testing_batch:tf.data.Dataset,**kwargs):
        X, y = testing_batch
        classes, coords = self.model(X, training=False)


        batch_cat_loss = self.cat_loss(tf.cast(y[0].numpy(),tf.float32).numpy(),classes)
        batch_local_loss = self.local_loss(tf.cast(y[1].numpy(),tf.float32),coords)

        total_loss = batch_local_loss + (0.5 * batch_cat_loss)

        logger.info(f"<<<< Testing Step: {'total_loss':total_loss, 'catagorical_loss':batch_cat_loss, 'localized_loss':batch_local_loss} >>>>")
        return {"total_loss":total_loss, "catagorical_loss":batch_cat_loss, "localized_loss":batch_local_loss}

    def call(self, X, **kwargs):
        return self.model(X,**kwargs)
    

if __name__=="__main__":
    __all__=["FaceTracker"]