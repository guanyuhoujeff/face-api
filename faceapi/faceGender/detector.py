import numpy as np
from faceapi.base import ModelHandler
from typing import  List
import torch
import cv2
from torchvision import transforms


class UTKfaceGenderTorchDetector(ModelHandler):
    def __init__(self, 
                 torch_model_path : str,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()

        self._torch_model_path = torch_model_path
        self._device = device
        
        self._transforms_inference = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
        self.initialize()


    def initialize (self,):
        if not self.initialized:
            
            self._model = torch.load(self._torch_model_path)
            self._model.to(self._device)
            self._model.eval()
            
            self.initialized = True


    def preprocess(self, face_img_list: List[np.ndarray]) -> torch.Tensor:
        """
        如何前處理資料
        """
        return [self._transforms_inference(face_img)[None].to(self._device) for face_img in face_img_list]
    
    def inference(self, model_input: List[torch.Tensor]) -> np.ndarray:
        """_summary_

        Args:
            model_input List[torch.Tensor]: _description_
            conf (float, optional): _description_. Defaults to 0.5.
            iou (float, optional): _description_. Defaults to 0.7.

        Returns:
            np.ndarray: _description_
        """
        res_list = []
        for data_to_model in model_input:
            # predict Age
            outputs = self._model(data_to_model)
            _, preds = torch.max(outputs, 1)
            pred = preds.cpu().numpy()[0]
            label =  ['female', 'male'][pred]
            res_list.append(label)
      
        # print('predicts', )
        # print(predicts)
        return np.array(res_list)

    def handle(self, img: np.ndarray) -> np.ndarray:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)





class CaffeDetector(ModelHandler):
    def __init__(self, 
                 age_proto_path = 'age_deploy.prototxt', 
                 age_model_path = 'age_net.caffemodel', 
                 
                 gender_proto_path = 'gender_deploy.prototxt', 
                 gender_model_path = 'gender_net.caffemodel', 
                 
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super().__init__()

        self._age_proto_path = age_proto_path
        self._age_model_path = age_model_path
        self._gender_proto_path = gender_proto_path
        self._gender_model_path = gender_model_path
        
        self._device = device
        
        self.age_list = ['(0-3)', '(4-7)', '(8-13)', '(14-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genders = ["Male", "Female"]
        
        self.initialize()


    def initialize (self,):
        if not self.initialized:
            
            self._age_model = cv2.dnn.readNet(self._age_model_path, self._age_proto_path)
            self._gender_model = cv2.dnn.readNet(self._gender_model_path, self._gender_proto_path)
            
            self.initialized = True

    def preprocess(self, face_img_list: List[np.ndarray]) -> torch.Tensor:
        """
        如何前處理資料
        """
        return [cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False) for face in face_img_list]
    
    def inference(self, model_input: List[np.ndarray]) -> np.ndarray:
        """_summary_

        Args:
            model_input (np.ndarray): _description_
            conf (float, optional): _description_. Defaults to 0.5.
            iou (float, optional): _description_. Defaults to 0.7.

        Returns:
            np.ndarray: _description_
        """
        res_list = []
        for face in model_input:
            # predict Age
            self._age_model.setInput(face)  # pass the 227x227 face blob to the age net.
            agePreds = self._age_model.forward()  # do forward pass
            age = self.age_list[agePreds[0].argmax()]  # get the age range
            
            # predict Gender
            self._gender_model.setInput(face)  # pass the 227x227 reshaped face blob to the gender net for prediction
            genderPreds = self._gender_model.forward()  # do the forward pass
            gender = self.genders[genderPreds[0].argmax()]  # get the gender

            res_list.append((age, gender))
      
        # print('predicts', )
        # print(predicts)
        return np.array(res_list)

    def handle(self, img: np.ndarray) -> np.ndarray:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

