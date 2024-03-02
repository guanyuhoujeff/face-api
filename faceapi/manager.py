import numpy as np
from typing import Union
import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    # pip install typing_extensions
    from typing_extensions import Literal
from typing import List 
from .base import FaceDetection, ModelHandler
from .faceDetector import Detector
from .faceEmbeddimg import Encoder
import os

# face detector
DETECTOR_MODEL_LIST = [
    # torch
    'torch-retinaface',
    'face-yolov8-n',
    'face-yolov8-s',
    'face-yolov8-m',
    'face-yolov8-l',
]

# face encoder
ENCODER_MODEL_LIST = [
    # torch
    'torch-vgg-face', ## 中量
    'inception-resnet-v1', ## 'vggface2' 輕量
]

# Union[Literal[
#                     'torch-retinaface', 
#                     'face-yolov8-n',
#                     'face-yolov8-s',
#                     'face-yolov8-m',
#                     'face-yolov8-l',], None]


class FaceAPIManager(ModelHandler):
    def __init__(self, 
                 detector_model_path = 'face-yolov8-m.pt',
                 encoder_model_path  = 'inception-resnet-v1.pt') -> None:
        super().__init__()
        
        self._detector_model_path = detector_model_path
        self._encoder_model_path = encoder_model_path
        
        self.initialize()

    def initialize(self):
        if not self.initialized:
            if os.path.isfile(self._detector_model_path):
                self._detector = Detector(self._detector_model_path)
            else:
                raise ValueError('Detector model {self._detector_model_path}, not found!')
            
            if os.path.isfile(self._encoder_model_path):
                self._encoder = Encoder(self._encoder_model_path)
            else:
                raise ValueError('Encoder model {self._encoder_model_path}, not found!')

            self.initialized = True

    def inference(self, img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        如何將前處理完的資料進行ai預測
        """
        
        faces = self._detector.handle(img, conf=conf, iou=iou)
        if not self._encoder is None and len(faces)>0:
            face_img_list = []
            for face in faces:
                x1,y1,x2,y2  = face.xyxy
                face_img = img[y1:y2,x1:x2,:]
                face_img_list.append(face_img)

            embeddings = self._encoder.handle(face_img_list)
            for face, embedding in zip(faces, embeddings):
                face.embedding = embedding

        return faces

    def handle(self,  img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input, conf, iou)
        return self.postprocess(model_output)