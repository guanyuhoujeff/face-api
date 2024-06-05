import numpy as np
from typing import List 
from .base import FaceDetection, ModelHandler

from .faceDetector import TorchDetector
from .faceEmbeddimg import TorchEncoder
from .faceEmotion import RafEmotionTorchDetector
from .faceAge import UTKfaceAgeTorchDetector
from .faceGender import UTKfaceGenderTorchDetector

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
                 age_model_path  = None, 
                 gender_model_path   = None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                 ) -> None:
        super().__init__()
        
        self._detector_model_path = detector_model_path
        
        self._encoder_model_path = encoder_model_path
        self._emotion_model_path = emotion_model_path
        self._age_model_path = age_model_path
        self._gender_model_path = gender_model_path
        
        
        self._device  = device
        # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.initialize()

    def initialize(self):
        if not self.initialized:
            
            # 人臉偵測(畫框框)
            if os.path.isfile(self._detector_model_path):
                self._detector = TorchDetector(self._detector_model_path, self._device)
            else:
                raise ValueError(f'Detector model {self._detector_model_path}, not found!')
            
            # 人臉辨識(Vip)
            if not self._encoder_model_path is None:
                if os.path.isfile(self._encoder_model_path):
                    self._encoder = TorchEncoder(self._encoder_model_path, self._device)
                else:
                    raise ValueError(f'encoder model {self._encoder_model_path}, not found!')
                
            # 情緒(Emotion)
            if not self._emotion_model_path is None:
                if os.path.isfile(self._emotion_model_path):
                    self._emotion_dector = RafEmotionTorchDetector(self._emotion_model_path, self._device)
                else:
                    raise ValueError(f'emotion model {self._emotion_model_path}, not found!')
            
            # 性別(Gender)
            if not self._gender_model_path is None:
                if os.path.isfile(self._gender_model_path):
                    self._gender_detector = UTKfaceGenderTorchDetector(self._gender_model_path, self._device)
                else:
                    raise ValueError(f'gender model {self._gender_model_path}, not found!')
            
            # 年紀(Age)
            if not self._age_model_path is None:
                if os.path.isfile(self._age_model_path):
                    self._age_detector = UTKfaceAgeTorchDetector(self._age_model_path, self._device)
                else:
                    raise ValueError(f'encoder model {self._age_model_path}, not found!')

            self.initialized = True

    def inference(self, img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        如何將前處理完的資料進行ai預測
        """
        
        faces = self._detector.handle(img, conf=conf, iou=iou)
        
        
        
        if len(faces)>0:
            # 如果用 cv2讀取影像，預設是BGR，要放入 resnet 的trans fun，則把資料轉成 RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_face_img_list = []
            face_img_list = []
            
            for face in faces:
                x1,y1,x2,y2  = face.xyxy
                face_img_list.append(img[y1:y2,x1:x2,:])
                rgb_face_img_list.append(rgb_img[y1:y2,x1:x2,:])

            ## vip
            if not self._encoder_model_path is None:
                embeddings = self._encoder.handle(rgb_face_img_list)
                for face, embedding in zip(faces, embeddings):
                    face.embedding = embedding
            
            ## 情緒
            if not self._emotion_model_path is None:
                emotions = self._emotion_dector.handle(rgb_face_img_list)
                for face, emotion in zip(faces, emotions):
                    face.emotion = emotion

            # age
            if not self._age_model_path is None:
                age_list = self._age_detector.handle(rgb_face_img_list)
                for face, age in zip(faces, age_list):
                    face.age = age
                    
            # gender
            if not self._gender_model_path is None:
                gender_list = self._gender_detector.handle(rgb_face_img_list)
                for face, gender in zip(faces, gender_list):
                    face.gender = gender
                    
            # ## age_gender
            # if all(
            #     [self._age_proto_path, 
            #      self._age_model_path,
            #      self._gender_proto_path,
            #      self._gender_model_path
            #     ]):

                    
            #     age_genders = self._age_gender_dector.handle(face_img_list)
            #     for face, age_gender in zip(faces, age_genders):
            #         face.age = age_gender[0]
            #         face.gender = age_gender[1]
                    

        return faces

    def handle(self,  img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input, conf, iou)
        return self.postprocess(model_output)

from .faceDetector import OpenvinoDetector
from .faceEmbeddimg import OpenvinoEncoder
from .faceEmotion import RafEmotionOpenvinoDetector
from .faceAge import UTKfaceAgeOpenvinoDetector
from .faceGender import UTKfaceGenderOpenvinoDetector




class OpenvinoFaceAPIManager(ModelHandler):
    def __init__(self, 
                 detector_model_path = 'face-yolov8-m.xml',
                 encoder_model_path  = None,
                 emotion_model_path  = None,
                 age_model_path  = None, 
                 gender_model_path   = None,
                 device= "CPU"
                 ) -> None:
        super().__init__()
        
        self._detector_model_path = detector_model_path
        
        self._encoder_model_path = encoder_model_path
        self._emotion_model_path = emotion_model_path
        self._age_model_path = age_model_path
        self._gender_model_path = gender_model_path
        
        self._device  = device
        # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.initialize()

    def initialize(self):
        if not self.initialized:
            
            # 人臉偵測(畫框框)
            if os.path.isfile(self._detector_model_path):
                self._detector = OpenvinoDetector(model_path= self._detector_model_path, device=self._device)
            else:
                raise ValueError(f'Detector model {self._detector_model_path}, not found!')
            
            # 人臉辨識(Vip)
            if not self._encoder_model_path is None:
                if os.path.isfile(self._encoder_model_path):
                    self._encoder = OpenvinoEncoder(self._encoder_model_path, self._device)
                else:
                    raise ValueError(f'encoder model {self._encoder_model_path}, not found!')
                
            # 情緒(Emotion)
            if not self._emotion_model_path is None:
                if os.path.isfile(self._emotion_model_path):
                    self._emotion_dector = RafEmotionOpenvinoDetector(self._emotion_model_path, self._device)
                else:
                    raise ValueError(f'emotion model {self._emotion_model_path}, not found!')
            
            # 性別(Gender)
            if not self._gender_model_path is None:
                if os.path.isfile(self._gender_model_path):
                    self._gender_detector = UTKfaceGenderOpenvinoDetector(self._gender_model_path, self._device)
                else:
                    raise ValueError(f'gender model {self._gender_model_path}, not found!')
            
            # 年紀(Age)
            if not self._age_model_path is None:
                if os.path.isfile(self._age_model_path):
                    self._age_detector = UTKfaceAgeOpenvinoDetector(self._age_model_path, self._device)
                else:
                    raise ValueError(f'encoder model {self._age_model_path}, not found!')

            self.initialized = True

    def inference(self, img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        如何將前處理完的資料進行ai預測
        """
        
        faces = self._detector.handle(img, conf=conf, iou=iou)
        
        
        
        if len(faces)>0:
            # 如果用 cv2讀取影像，預設是BGR，要放入 resnet 的trans fun，則把資料轉成 RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_face_img_list = []
            face_img_list = []
            
            for face in faces:
                x1,y1,x2,y2  = face.xyxy
                face_img_list.append(img[y1:y2,x1:x2,:])
                rgb_face_img_list.append(rgb_img[y1:y2,x1:x2,:])

            ## vip
            if not self._encoder_model_path is None:
                embeddings = self._encoder.handle(rgb_face_img_list)
                for face, embedding in zip(faces, embeddings):
                    face.embedding = embedding
            
            ## 情緒
            if not self._emotion_model_path is None:
                emotions = self._emotion_dector.handle(rgb_face_img_list)
                for face, emotion in zip(faces, emotions):
                    face.emotion = emotion

            # age
            if not self._age_model_path is None:
                age_list = self._age_detector.handle(rgb_face_img_list)
                for face, age in zip(faces, age_list):
                    face.age = age
                    
            # gender
            if not self._gender_model_path is None:
                gender_list = self._gender_detector.handle(rgb_face_img_list)
                for face, gender in zip(faces, gender_list):
                    face.gender = gender
                    
            # ## age_gender
            # if all(
            #     [self._age_proto_path, 
            #      self._age_model_path,
            #      self._gender_proto_path,
            #      self._gender_model_path
            #     ]):

                    
            #     age_genders = self._age_gender_dector.handle(face_img_list)
            #     for face, age_gender in zip(faces, age_genders):
            #         face.age = age_gender[0]
            #         face.gender = age_gender[1]
                    

        return faces

    def handle(self,  img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input, conf, iou)
        return self.postprocess(model_output)