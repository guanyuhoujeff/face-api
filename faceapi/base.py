from pydantic import BaseModel
from typing import List, Optional
from abc import ABC, abstractmethod

class FaceDetection(BaseModel):
    xyxy: List[int] = []
    score: float = 0.0
    landms: List[float] = []
    embedding: List[float] = []
    emotion: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None


class ModelHandler(ABC):
    """
    此架構參考 torch serve 的 ModelHandler
    """

    def __init__(self):
        self.initialized = False

    @abstractmethod
    def initialize(self):
        """
        初始化模型
        """
        pass  # 或者抛出 NotImplementedError 异常

    def preprocess(self, data):
        """
        如何前處理資料
        """
        return data

    def inference(self, model_input):
        """
        如何將前處理完的資料進行ai預測
        """
        return model_input

    def postprocess(self, inference_output):
        """
        如何將ai預測完的資料做後續處理
        """
        return inference_output

    def handle(self, data):
        """
        整個ai辨識流程，從原始資料->前處理->ai預測->後續處理
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)