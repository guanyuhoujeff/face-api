import numpy as np
from faceapi.base import ModelHandler
from typing import  List
import torch
from .networks.DDAM import DDAMNet
from torchvision import transforms
import cv2

class TorchDetector(ModelHandler):
    def __init__(self, model_path = 'face-emotion.pth', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()

        self._model_path = model_path
        self._device = device
        
        self.initialize()


    def _buildTransform(
        self,
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]) -> torch.Tensor:
        self.transform = transforms.Compose([
            lambda img : cv2.resize(img, (112, 112)),  ## 因為目前 embedding 訓練的模型是 160*160
            lambda img : np.float32(img)/255,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def initialize (self,):
        if not self.initialized:
            self._model =  DDAMNet(num_class=7,num_head=2)
            checkpoint = torch.load(self._model_path, map_location=self._device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.to(self._device)
            self._model.eval()  
            self._buildTransform()
            self.initialized = True

    def preprocess(self, face_img_list: List[np.ndarray]) -> torch.Tensor:
        """
        如何前處理資料
        """
        return torch.stack([self.transform(im) for im in face_img_list]).to(self._device)
    
    def inference(self, model_input: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            model_input (np.ndarray): _description_
            conf (float, optional): _description_. Defaults to 0.5.
            iou (float, optional): _description_. Defaults to 0.7.

        Returns:
            np.ndarray: _description_
        """
        with torch.no_grad():
            outs, feats, heads = self._model(model_input)
        predicts = [self._model.labels[out.argmax()] for out in outs.cpu().numpy()]
        # print('predicts', )
        # print(predicts)
        return np.array(predicts)

    def postprocess(self, inference_output):
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

