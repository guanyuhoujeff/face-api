from faceapi.base import ModelHandler
import torch
import numpy as np
from torchvision import transforms
from typing import  List
from .facenet.models.inception_resnet_v1 import InceptionResnetV1
from .facenet.models.mtcnn import fixed_image_standardization
import cv2

import openvino as ov
import json

class TorchEncoder(ModelHandler):
    def __init__(self, model_path=None, device=None) -> None:
        """_summary_
        
        此模型的embedding 部分，主要參考
        https://github.com/timesler/facenet-pytorch

        Args:
            model_path (_type_, optional): 如果有提供 finetune 完的模型路徑，則會以讀取該模型(輸出會是該模型的分類)。如果沒有，就會走預設的(輸出會是512個embedding). Defaults to None.
        """
        super().__init__()
        
        self._model_path = model_path
        self._device = device
        self.class_to_idx = {}

        self.initialize()
        
    def _buildTransform(self):
        self.transform = transforms.Compose([
            lambda img : cv2.resize(img, (160,160)),  ## 因為目前 embedding 訓練的模型是 160*160
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        
        
    def initialize(self,):
        if not self.initialized:
            # 模型走官方預設pretrain 的權重
            self._model = InceptionResnetV1(
                classify=False,
                pretrained='vggface2'
            )
            if not self._model_path is None:
                # 如果有finetune過的，即可模型走官方預設pretrain 的權重
                self._model.loadFinetuneModel(self._model_path)
                self.class_to_idx = self._model.class_to_idx
                
            self._model.to(self._device)
            self._model.eval()
            self._buildTransform()
            self.initialized = True

    def preprocess(self, face_img_list: List[np.ndarray]) -> torch.Tensor:
        """
        如何前處理資料
        """
        return torch.stack([self.transform(im) for im in face_img_list]).to(self._device)
    
    def inference(self, model_input: torch.Tensor) ->  torch.Tensor:
        """_summary_

        Args:
            model_input (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        face_embeddings = self._model(model_input)
        return face_embeddings

    def postprocess(self, inference_output: torch.Tensor) -> np.ndarray:
        """
        如何將ai預測完的資料做後續處理
        """
        return inference_output.cpu().detach().numpy()

    def handle(self, img: np.ndarray) -> np.ndarray:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
    
    
    
class OpenvinoEncoder(ModelHandler):
    def __init__(self, model_path=None, device=None) -> None:
        """_summary_
        
        此模型的embedding 部分，主要參考
        https://github.com/timesler/facenet-pytorch

        Args:
            model_path (_type_, optional): 如果有提供 finetune 完的模型路徑，則會以讀取該模型(輸出會是該模型的分類)。如果沒有，就會走預設的(輸出會是512個embedding). Defaults to None.
        """
        super().__init__()
        self._core = ov.Core()
        self._model_path = model_path
        self._meta_path = model_path.replace('.xml', '.meta')
        self.class_to_idx = {}


        self._device = device
        
        self.initialize()
        
    def _buildTransform(self):
        self.transform = transforms.Compose([
            lambda img : cv2.resize(img, (160,160)),  ## 因為目前 embedding 訓練的模型是 160*160
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        
        
    def initialize (self,):
        
        if not self.initialized:
            with open(self._meta_path, 'r') as reader:
                self.class_to_idx = json.loads(reader.read())
                
            model_ir = self._core.read_model(model=self._model_path)
            self._model = self._core.compile_model(model=model_ir, device_name=self._device)
            self._buildTransform()
            self.initialized = True

    def preprocess(self, face_img_list: List[np.ndarray]) -> np.ndarray:
        """
        如何前處理資料
        """
        normalized_input_image = torch.stack([self.transform(im) for im in face_img_list]).numpy()
        return normalized_input_image
    
    def inference(self, model_input: np.ndarray) ->  np.ndarray:
        """_summary_

        Args:
            model_input (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        face_embeddings = self._model(model_input)[self._model.output(0)]
        return face_embeddings

    def postprocess(self, inference_output: np.ndarray) -> np.ndarray:
        """
        如何將ai預測完的資料做後續處理
        """
        return inference_output

    def handle(self, img: np.ndarray) -> np.ndarray:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)