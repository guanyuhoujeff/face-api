import numpy as np
from typing import List 
from .base import FaceDetection, ModelHandler
from .faceDetector import OpenvinoDetector, TorchDetector
from .faceEmbeddimg import OpenvinoEncoder, TorchEncoder

from .faceEmotion import TorchDetector as EmotionHandler
from .faceAgeGender import CaffeDetector as AgeGenderHandler

import os
import cv2
import torch

# face detector
DETECTOR_MODEL_LIST = [
    # torch
    'torch-retinaface',
    'face-yolov8-n',
    'face-yolov8-s',
    'face-yolov8-m',
    'face-yolov8-l',
]

# Union[Literal[
#'torch-retinaface', 
#'face-yolov8-n',
#'face-yolov8-s',
#'face-yolov8-m',
#'face-yolov8-l',], None]


class FaceAPIManager(ModelHandler):
    def __init__(self, 
                 detector_model_path = 'face-yolov8-m.pt',
                 
                 encoder_model_path  = None,
                 emotion_model_path  = None,
                 
                 age_proto_path = 'age_deploy.prototxt', 
                 age_model_path = 'age_net.caffemodel', 
                 
                 gender_proto_path = 'gender_deploy.prototxt', 
                 gender_model_path = 'gender_net.caffemodel', 
                 
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                 ) -> None:
        super().__init__()
        
        self._detector_model_path = detector_model_path
        self._encoder_model_path = encoder_model_path
        self._emotion_model_path = emotion_model_path
        
        self._age_proto_path = age_proto_path
        self._age_model_path = age_model_path
        self._gender_proto_path = gender_proto_path
        self._gender_model_path = gender_model_path
        
        
        self._device  = device
        # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.initialize()

    def initialize(self):
        if not self.initialized:
            if os.path.isfile(self._detector_model_path):
                if '.xml' in self._detector_model_path:
                    ## 走 openvino
                    if self._device is None:
                        ## 如果沒給 device 就預設cpu
                        self._device = "CPU"
                    self._detector = OpenvinoDetector(self._detector_model_path, self._device)
                else:
                    self._detector = TorchDetector(self._detector_model_path, self._device)
            else:
                raise ValueError(f'Detector model {self._detector_model_path}, not found!')
            
            
            if '.xml' in self._encoder_model_path:
                ## 走 openvino
                if self._device is None:
                    self._device = "CPU"
                self._encoder = OpenvinoEncoder(self._encoder_model_path, self._device)
            else:
                # 假如 self._encoder_model_path 是None，則會讀取預設的官方權重
                self._encoder = TorchEncoder(self._encoder_model_path, self._device)
                
            # emotion
            if not self._emotion_model_path is None:
                self._emotion_dector = EmotionHandler(self._emotion_model_path, self._device)
            
                
            ## age gender 
            if all(
                [self._age_proto_path, 
                 self._age_model_path,
                 self._gender_proto_path,
                 self._gender_model_path
                ]):
                self._age_gender_dector = AgeGenderHandler(
                    age_proto_path = self._age_proto_path, 
                    age_model_path= self._age_model_path, 
                    gender_proto_path=self._gender_proto_path, 
                    gender_model_path=self._gender_model_path, 
                    device= self._device
                )
                
            
            
            self.initialized = True

    def inference(self, img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        如何將前處理完的資料進行ai預測
        """
        
        faces = self._detector.handle(img, conf=conf, iou=iou)
        if not self._encoder is None and len(faces)>0:
            # 如果用 cv2讀取影像，預設是BGR，要放入 resnet 的trans fun，則把資料轉成 RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_face_img_list = []
            face_img_list = []
            
            for face in faces:
                x1,y1,x2,y2  = face.xyxy
                face_img_list.append(img[y1:y2,x1:x2,:])
                rgb_face_img_list.append(rgb_img[y1:y2,x1:x2,:])

            embeddings = self._encoder.handle(rgb_face_img_list)
            for face, embedding in zip(faces, embeddings):
                face.embedding = embedding
            
            ## 情緒
            if not self._emotion_model_path is None:
                emotions = self._emotion_dector.handle(rgb_face_img_list)
                for face, emotion in zip(faces, emotions):
                    face.emotion = emotion
                    
            ## age_gender
            if all(
                [self._age_proto_path, 
                 self._age_model_path,
                 self._gender_proto_path,
                 self._gender_model_path
                ]):

                    
                age_genders = self._age_gender_dector.handle(face_img_list)
                for face, age_gender in zip(faces, age_genders):
                    face.age = age_gender[0]
                    face.gender = age_gender[1]
                    

        return faces

    def handle(self,  img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input, conf, iou)
        return self.postprocess(model_output)