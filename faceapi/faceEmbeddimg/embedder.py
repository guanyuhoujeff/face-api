from faceapi.base import ModelHandler
import torch
import numpy as np
from torchvision import transforms
from typing import  List
# from .inception_resnet import InceptionResnetV1
from .facenet.models.resnet import Resnet34Triplet


class Encoder(ModelHandler):
    """_summary_
    
    the face embedding model of this project is using facenet-pytorch-glint360k
    https://github.com/tamerthamoqa/facenet-pytorch-glint360k/tree/master?tab=readme-ov-file
    
    """
    def __init__(self, model_path ) -> None:
        super().__init__()
        
        self._model_path = model_path
        self.initialize()
        
    def _buildTransform(self):
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((140, 140)),  # Pre-trained model uses 140x140 input images
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.6071, 0.4609, 0.3944],  # Normalization settings for the model, the calculated mean and std values
                    std=[0.2457, 0.2175, 0.2129]     # for the RGB channels of the tightly-cropped glint360k face dataset
            )
        ])
        
        
    def initialize (self,):
        self._devie = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if not self.initialized:
            
            checkpoint = torch.load(self._model_path, map_location=self._devie)
            self._model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.to(self._devie)
            self._model.eval()
            
            self._buildTransform()
            
            self.initialized = True

    def preprocess(self, face_img_list: List[np.ndarray]) -> torch.Tensor:
        """
        如何前處理資料
        """
        return torch.stack([ self.transform(im) for im in face_img_list]).to(self._devie)
    
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