
import os, shutil
import tensorflow as tf
from src.facedetection_tf.logger import FD_Logger
import numpy as np
from typing import List, Tuple
from tqdm.auto import tqdm

logger = FD_Logger().logger()


def train_test_split_indices(img_folder_path:os.path.dirname, lab_folder_path:os.path, train_len:float, 
                             val_len:float, test_len:float, ds:tf.data.Dataset):

    img_files = os.listdir(img_folder_path)
    lab_files = os.listdir(lab_folder_path)

    num_images=len(img_files)
    train_len = int(len(ds) * train_len)
    val_len = int(len(ds) * val_len)
    test_len = int(len(ds) * test_len)

    train_idx=[]
    test_idx=[]
    val_idx=[]

    for i,_  in tqdm(enumerate(img_files), total=num_images):
        idx=np.random.randint(low=0,high=num_images)
        while ((idx in train_idx) or (idx in test_idx) or (idx in val_idx)) and (num_images>=(len(train_idx)+len(test_idx)+len(val_idx))):
            idx=np.random.randint(low=0,high=num_images)
        if len(train_idx)<train_len:
            train_idx.append((idx, img_files[idx], lab_files[idx]))
        elif len(val_idx)<val_len:
            val_idx.append((idx, img_files[idx], lab_files[idx]))
        else:
            test_idx.append((idx, img_files[idx], lab_files[idx]))
    logger.info("<<<< train_test_split_indices generated for images and labels >>>>")
    return train_idx, val_idx, test_idx


def create_partitions(dst_folder:str, src_folder:os.path, sequence:List[Tuple[int, str, str]]):
    """
    folder_name: its usually the data folder. Please maintain this structure
    ```
    * data -> train -> images
                    -> labels
           -> test ...
           -> val ...
    
    source_folder : 
    * data -> images
           -> labels
    ```
    """

    img_dir_path = os.path.join("data",dst_folder, "images")
    lab_dir_path = os.path.join("data",dst_folder, "labels")

    if not (os.path.exists(img_dir_path)) or (os.path.exists(lab_dir_path)):
        os.makedirs(img_dir_path)
        os.makedirs(lab_dir_path)

    for file in sequence:
        img_file_name=file[1]
        lab_file_name=file[-1]
        shutil.copy(src=os.path.join(src_folder,"images", img_file_name), dst=os.path.join(img_dir_path,img_file_name))
        shutil.copy(src=os.path.join(src_folder,"labels", lab_file_name), dst=os.path.join(lab_dir_path,lab_file_name))

if __name__=="__main__":
    __all__=["train_test_split_indices", "create_partitions"]