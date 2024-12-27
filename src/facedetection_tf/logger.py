import logging
from logging import StreamHandler, FileHandler
import os
from datetime import datetime
import sys
from pathlib import Path
from dataclasses import dataclass, field

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y')}.log"

LOG_PATH=os.path.join(os.getcwd(),"logs",LOG_FILE)

os.makedirs(os.path.dirname(LOG_PATH),exist_ok=True)

@dataclass
class FD_Logger:

    LOG_PATH:os.path=field(default=LOG_PATH)

    def __post_init__(self):
        self.stream_handler = StreamHandler(sys.stdout)
        self.file_handler = FileHandler(self.LOG_PATH)

    def logger(self):
        logging.basicConfig(
            #filename=LOG_FILE_PATH,
            format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                self.stream_handler,
                self.file_handler
            ]
        )

        return logging.getLogger('facedetection')

if __name__=="__main__":
    __all__=["FD_Logger"]