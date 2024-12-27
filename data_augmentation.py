import albumentations as alb
import os, sys, shutil
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm.auto import tqdm
from src.facedetection_tf.exception import FD_Exception
from src.facedetection_tf.logger import FD_Logger

from typing import Any, List, Tuple

logger = FD_Logger().logger()

augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)],
    bbox_params=alb.BboxParams(format='albumentations',label_fields=['class_labels'])
)



def get_augments(augmentor:alb.Compose, image:Any, label:Any, class_label:str, max_coords:List[int]=[640,480,640,480]):
    """
    eg:
    >>> cv2.imread("data/train/images/a0b7accf-c15b-11ef-a853-38fc980eedf8.jpg")
    >>> with open("data/train/labels/a0b7accf-c15b-11ef-a853-38fc980eedf8.json","r") as handle:
    >>>   label = json.load(handle)
    """
    coords = [0,0,0,0]

    coords[0]=label['shapes'][0]['points'][0][0]
    coords[1]=label['shapes'][0]['points'][0][1]
    coords[2]=label['shapes'][0]['points'][1][0]
    coords[3]=label['shapes'][0]['points'][1][1]

    coords = list(np.divide(coords,max_coords))

    return augmentor(image=image,bboxes=[coords], class_labels=[class_label])


if __name__=="__main__":
    try:
        for folder in ["train","test","val"]:
            dir_name = 'data'
            dst_dir_name = 'augmented_data'
            folder_name = os.path.join(dir_name,folder)
            
            img_folder = os.path.join(folder_name,"images")
            lab_folder = os.path.join(folder_name,"labels")

            img_augmented_folder = os.path.join(dir_name,dst_dir_name,folder,"images")
            lab_augmented_folder = os.path.join(dir_name,dst_dir_name,folder,"labels")

            

            for image_file_name in tqdm(os.listdir(img_folder), desc=f"Folder: {folder}"):
                image = cv2.imread(os.path.join(img_folder,image_file_name))
                label_file_path = os.path.join(lab_folder,image_file_name.replace(".jpg",".json"))
                with open(label_file_path,"r") as handle:
                    label = json.load(handle)

                if not (os.path.exists(img_augmented_folder)) or (os.path.exists(lab_augmented_folder)):
                    os.makedirs(img_augmented_folder, exist_ok=True)
                    os.makedirs(lab_augmented_folder, exist_ok=True)
                
                annotations={}
            ## This for loop for creating multiple augmentations for each image
                annotations["image_name"]=image_file_name
                for x in range(120):

                    augmented = get_augments(augmentor=augmentor, image=image, label=label, class_label='face')
                    if len(augmented["bboxes"]) == 0:
                        annotations["bbox"] = dict(zip([0,1,2,3],[0,0,0,0]))
                        annotations["class_label"] = 0

                    else:
                        annotations["bbox"] = dict(zip([0,1,2,3],augmented["bboxes"][0]))
                        annotations["class_label"] = 1 #augmented["class_labels"]


                            #################### SAVE AUGMENTED IMAGES AND LABELS #############################
                    file_name = image_file_name.replace(".jpg",f"_{x}")
                    cv2.imwrite(filename=os.path.join(img_augmented_folder,f"{file_name}.jpg"), img=augmented['image'])
                    with open(os.path.join(lab_augmented_folder,f"{file_name}.json"), 'w') as handle:
                        json.dump(annotations,handle)
            logger.info(f"<<<< Data Augmentation successfully completed for {folder} images and labels and stored in {folder_name}")
    except Exception as e:
        logger.info(FD_Exception(e,sys))
        raise FD_Exception(e,sys)