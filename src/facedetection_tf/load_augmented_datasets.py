import tensorflow as tf
from src.facedetection_tf.utils import load_image, load_labels
from src.facedetection_tf.logger import FD_Logger
import json
import os, sys
from dataclasses import dataclass, field

logger = FD_Logger().logger()

@dataclass
class generate_datasets:

    aug_train_lab_folder:os.path=field(default=r"data/augmented_data/train/labels/*.json")
    aug_test_lab_folder:os.path=field(default=r"data/augmented_data/test/labels/*.json")
    aug_val_lab_folder:os.path=field(default=r"data/augmented_data/val/labels/*.json")

    aug_train_img_folder_path:os.path=field(default=r"data/augmented_data/train/images/*.jpg")
    aug_test_img_folder_path:os.path=field(default= r"data/augmented_data/test/images/*.jpg")
    aug_val_img_folder_path:os.path=field(default=r"data/augmented_data/val/images/*.jpg")

    def get_image_datasets(self):
        aug_images_ds_gathered=[]
        for aug_images_ds in [self.aug_train_img_folder_path, self.aug_test_img_folder_path, self.aug_val_img_folder_path]:
            aug_images_ds = tf.data.Dataset.list_files(aug_images_ds, shuffle=False)
            aug_images_ds = aug_images_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            aug_images_ds = aug_images_ds.map(lambda x: tf.image.resize(x, (120,120)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            aug_images_ds = aug_images_ds.map(lambda x: x/255, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            aug_images_ds_gathered.append(aug_images_ds) 
        logger.info("<<<< Train_augmented images loaded and intialized >>>>")
        return aug_images_ds_gathered
    
    @tf.autograph.experimental.do_not_convert
    def get_label_datasets(self):
        aug_labels_ds_gathered=[]
        for aug_labels_ds in [self.aug_train_lab_folder, self.aug_test_lab_folder, self.aug_val_lab_folder]:
            aug_labels_ds = tf.data.Dataset.list_files(aug_labels_ds, shuffle=False)
            aug_labels_ds = aug_labels_ds.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #aug_labels_ds = aug_labels_ds.map(lambda x, y: (tf.convert_to_tensor(x),tf.convert_to_tensor(y)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            aug_labels_ds_gathered.append(aug_labels_ds)
        logger.info("<<<< Annoted labels loaded and intialized >>>>")
        return aug_labels_ds_gathered

    def generate_final_datasets(self):
        gathered_final_ds = []
        logger.info("<<<< Generating final datasets... >>>>")
        for image_ds, label_ds in zip(self.get_image_datasets(), self.get_label_datasets()):
            ds = tf.data.Dataset.zip(image_ds, label_ds)
            ds = ds.batch(8)
            #ds = ds.prefetch(4)
            gathered_final_ds.append(ds)
        
        logger.info("<<<< Final datasets generated... >>>>")
        return gathered_final_ds
    

if __name__=="__main__":
    __all__=["generate_datasets"]











    

