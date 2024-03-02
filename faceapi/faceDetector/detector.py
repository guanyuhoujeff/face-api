import torch
import torch.backends.cudnn as cudnn

from .model import cfg_re50, RetinaFace, load_model
from .utils import PriorBox, decode, decode_landm, py_cpu_nms

import numpy as np
import cv2
from ultralytics import YOLO
import os
from faceapi.base import FaceDetection, ModelHandler

from typing import  List

# class RetinaFaceDetector(ModelHandler):
#     先封掉，以yolo 為主
#     def __init__(self, model_pt_path, device :torch.device = None):
#         super(RetinaFace, self).__init__()

#         self.model_pt_path = model_pt_path

#         # Set device type
#         if torch.cuda.is_available(): self.device = device
#         else: self.device = torch.device("cpu")
#         print('!!!! device', self.device)

#     def _ndarrayToTensor(self, ndarray_img: np.ndarray):
#         img = np.float32(ndarray_img)
#         img -= (104, 117, 123)
#         img = img.transpose(2, 0, 1)
#         return torch.from_numpy(img)
    
#     def initialize(self,):
#         print('!!!! self.model_pt_path', self.model_pt_path)
#         self.model = self._load_torchscript_model(self.model_pt_path)
#         self.initialized = True

#     def _load_torchscript_model(self, model_pt_path):
#         """Loads the PyTorch model and returns the NN model object.

#         Args:
#             model_pt_path (str): denotes the path of the model file.

#         Returns:
#             (NN Model Object) : Loads the model object.
#         """
#         # TODO: remove this method if https://github.com/pytorch/text/issues/1793 gets resolved

#         model = RetinaFace(cfg=cfg_re50, phase = 'test')
#         model = load_model(model, model_pt_path, self.device.type == 'cpu')
#         model.eval()
#         cudnn.benchmark = True
#         model.to(self.device)
#         return model
    

#     def preprocess(self, data):
#         """The preprocess function of MNIST program converts the input data to a float tensor

#         Args:
#             data (List): Input data from the request is in the form of a Tensor

#         Returns:
#             list : The preprocess function returns the input image as a list of float tensors.
#         """
#         images = []
#         for row in data:
#             # Compat layer: normally the envelope should just return the data
#             # directly, but older versions of Torchserve didn't have envelope.
#             if isinstance(row, (np.ndarray)):
#                 image=  row
#             else:
#                 image = row.get("data") or row.get("body")


#             if isinstance(image, str):
#                 # if the image is a string of bytesarray.
#                 # image = base64.b64decode(image)
#                 raise ValueError('base64 未測試')
            
#             # If the image is sent as bytesarray
#             if isinstance(image, (bytearray, bytes)):
#                 # image = Image.open(io.BytesIO(image))
#                 # image = self.image_processing(image)
#                 np_arr = np.frombuffer(image, np.uint8)
#                 img_arr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#                 image = self._ndarrayToTensor(img_arr)
#             else:
#                 # if the image is a list
#                 # print(image.shape)
#                 image = self._ndarrayToTensor(image)
    

#             images.append(image)

#         return torch.stack(images).to(self.device)

#     def postprocess(self, res):
#         return res

#     def inference(self, model_input):
#         """
#         Internal inference methods
#         :param model_input: transformed model input data
#         :return: list of inference output in NDArray
#         """
#         # Do some inference call to engine here and return output
#         model_output = self.model.forward(model_input)
#         return model_output

#     def handle(self, data, nms_threshold = 0.4, confidence_threshold = 0.02):
#         """
#         Invoke by TorchServe for prediction request.
#         Do pre-processing of data, prediction using model and postprocessing of prediciton output
#         :param data: Input data for prediction
#         :param context: Initial context contains model server system properties.
#         :return: prediction output
#         """

#         model_input = self.preprocess(data)
#         im_height, im_width = model_input.shape[-2:]
#         model_output = self.inference(model_input)
#         loc, conf, landms = model_output
#         priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
#         priors = priorbox.forward()
#         priors = priors.to(self.device)
#         prior_data = priors.data


#         boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
#         boxes = boxes.cpu().numpy()


#         scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
#         landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
#         landms = landms.cpu().numpy()


#         # ignore low scores
#         inds = np.where(scores > confidence_threshold)[0]
#         boxes = boxes[inds]
#         landms = landms[inds]
#         scores = scores[inds]

#         # keep top-K before NMS
#         # order = scores.argsort()[::-1][:self.top_k]
#         order = scores.argsort()[::-1]
#         boxes = boxes[order]
#         landms = landms[order]
#         scores = scores[order]


#         # do NMS
#         dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#         keep = py_cpu_nms(dets, nms_threshold)

#         dets = dets[keep, :]
#         landms = landms[keep]


#         dets = np.concatenate((dets, landms), axis=1)

#         scale = np.array([
#             im_width, im_height, im_width, im_height, 1, 
#             im_width, im_height, im_width, im_height, im_width, im_height, im_width, im_height, im_width, im_height]
#         )
#         dets = dets * scale
#         dets = self.postprocess(dets)
#         return dets


class Detector(ModelHandler):
    def __init__(self, model_path = 'face-yolov8-m') -> None:
        super().__init__()

        self._model_path = model_path
        self.initialize()

    def initialize (self,):
        self._devie = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

