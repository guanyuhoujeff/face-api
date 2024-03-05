import numpy as np
from ultralytics import YOLO
from faceapi.base import FaceDetection, ModelHandler

from typing import  List
import openvino as ov
from .openvino_utils import (
    preprocess_image,
    image_to_tensor,
    postprocess
)




class TorchDetector(ModelHandler):
    def __init__(self, model_path = 'face-yolov8-m', device=None) -> None:
        super().__init__()

        self._model_path = model_path
        self._devie = device
        
        self.initialize()

    def initialize (self,):
        if not self.initialized:
            self._model = YOLO(self._model_path)
            self.initialized = True


    def inference(self, model_input: np.ndarray, conf=0.5, iou=0.7) -> np.ndarray:
        """_summary_

        Args:
            model_input (np.ndarray): _description_
            conf (float, optional): _description_. Defaults to 0.5.
            iou (float, optional): _description_. Defaults to 0.7.

        Returns:
            np.ndarray: _description_
        """
        img_h, img_w = model_input.shape[:2]
        res = self._model.predict(model_input , imgsz=(img_h, img_w), conf=conf , iou=iou, verbose=False)
        return res[0].boxes.data.cpu().numpy()

    def postprocess(self, inference_output) -> List[FaceDetection]:
        """
        如何將ai預測完的資料做後續處理
        """
        return [
            FaceDetection( 
                xyxy = [int(d) for d in output[:4]], 
                score = float(output[4])
            ) for output in inference_output
        ]


    def handle(self, img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input, conf, iou)
        return self.postprocess(model_output)




class OpenvinoDetector(ModelHandler):
    def __init__(self, model_path = 'face-yolov8-m', device=None) -> None:
        super().__init__()
        # Create OpenVINO Core object instance
        self._core = ov.Core()
        self._model_path = model_path
        self._devie = device
        
        self.initialize()

    def initialize (self,):
        if not self.initialized:
                        
            det_ov_model = self._core.read_model(self._model_path)
            if self._devie != "CPU":
                det_ov_model.reshape({0: [1, 3, 640, 640]})
            
            ov_config = {}
            if "GPU" in self._devie or ("AUTO" in self._devie and "GPU" in self._core.available_devices):
                ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
            self._model = self._core.compile_model(det_ov_model, self._devie, ov_config)
  
            self.initialized = True


    def inference(self, model_input: np.ndarray, conf=0.5, iou=0.7) -> np.ndarray:
        """_summary_

        Args:
            model_input (np.ndarray): _description_
            conf (float, optional): _description_. Defaults to 0.5.
            iou (float, optional): _description_. Defaults to 0.7.

        Returns:
            np.ndarray: _description_
        """
        
        preprocessed_image = preprocess_image(model_input)
        input_tensor = image_to_tensor(preprocessed_image)
        result = self._model(input_tensor)
        boxes = result[self._model.output(0)]
        input_hw = input_tensor.shape[2:]
        res = postprocess(
            pred_boxes=boxes, 
            min_conf_threshold=conf, 
            nms_iou_threshold=iou, 
            input_hw=input_hw, 
            orig_img=model_input, nc=1
        )
        print(res)
        det = res[0]['det']
        if len(det) >0:
            return det.numpy()
        else:
            return np.array([])


    def postprocess(self, inference_output) -> List[FaceDetection]:
        """
        如何將ai預測完的資料做後續處理
        """
        return [
            FaceDetection( 
                xyxy = [int(d) for d in output[:4]], 
                score = float(output[4])
            ) for output in inference_output
        ]


    def handle(self, img: np.ndarray, conf=0.5, iou=0.7) -> List[FaceDetection]:
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(img)
        model_output = self.inference(model_input, conf, iou)
        return self.postprocess(model_output)

