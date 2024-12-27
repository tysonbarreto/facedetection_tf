import os, time, cv2
from uuid import uuid1
from dataclasses import dataclass
from src.facedetection_tf.logger import FD_Logger
from src.facedetection_tf.exception import FD_Exception

logger = FD_Logger().logger()

@dataclass
class ImageCollector:

    number_images:int=30
    camer_no:int=0

    def __post_init__(self):
        self.IMAGES_PATH = os.path.join("data", "images")
        self.LABELS_PATH = os.path.join("data", "labels")
        if not (os.path.exists(self.IMAGES_PATH)) or (os.path.exists(self.IMAGES_PATH)):
            os.makedirs(self.IMAGES_PATH, exist_ok=True)
            os.makedirs(self.LABELS_PATH, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        logger.info(f"<<<< Path {self.IMAGES_PATH} intialized to capture {self.number_images} images >>>>")
    
    def capture(self):
        for imgnum in range(self.number_images):
            ret, frame = self.cap.read()
            imagename = os.path.join(self.IMAGES_PATH,f"{str(uuid1())}.jpg")
            cv2.imwrite(imagename, frame)
            cv2.imshow('frame', frame)
            time.sleep(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info(f"<<<< {self.number_images} images captured and stored in path {self.IMAGES_PATH} >>>>")

if __name__=="__main__":
    __all__ =["ImageCollector"]
