
import cv2
import os
import time
import matplotlib.pyplot as plt
import torch
import openvino as ov
from faceapi import OpenvinoFaceAPIManager
# from melo.api import TTS

core = ov.Core()
core.available_devices

THIS_DIR = os.path.dirname(__file__)

face_api_manager = OpenvinoFaceAPIManager(
    detector_model_path= os.path.join(THIS_DIR, "./model_weights/face-yolov8-m_openvino_model/face-yolov8-m.xml"),
    # encoder_model_path=r'C:\Users\jeffg\face-project\face-api\model_weights\feifei_face.pt',
    emotion_model_path= os.path.join(THIS_DIR,"./model_weights/face_emotion_openvino_model/face_emotion_resnet18.xml"), 
    age_model_path = os.path.join(THIS_DIR,"./model_weights/face_age_openvino_model/face_age_resnet18.xml"), 
    gender_model_path = os.path.join(THIS_DIR,"./model_weights/face_gender_openvino_model/face_gender_resnet18.xml"), 

    device= 'GPU'
)

print(1)

cap = cv2.VideoCapture("rtsp://172.23.200.68/live1.sdp")
# 初始化 FPS 计算
fps_start_time = time.time()
fps_frame_count = 0
fps = 0


print( cap.read())
while 1:
    d, frame = cap.read()
    if not d:
        print('no image')
        time.sleep(1)
        continue

    display_frame = frame.copy()
    faces = face_api_manager.handle(frame, conf=0.3)
    for face in faces:
        label = "None"
        pred_prob = 0.0
        if len(face.embedding)>0:
            pred_idx = face.embedding.argmax()
            # 如果爆掉，很大原因是因為不是finetune model
            pred_prob = face.embedding[pred_idx]
            label = {
               v: k
               for k , v in face_api_manager._encoder.class_to_idx.items()
            }[pred_idx]
            #print(label, pred_prob)
        
        x1,y1,x2,y2  = face.xyxy  
        #reid = reid_encoder.inference(img, np.array(face.xyxy)[None]).ravel()
        cv2.putText(display_frame, f"%s"%(face.emotion), (x1, y1),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        cv2.putText(display_frame, f"%s"%(face.gender), (x1, y1 + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        cv2.putText(display_frame, f"%s"%(face.age), (x1, y1 + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        cv2.rectangle(display_frame, ( x1,y1), (x2,y2), (255, 255, 255))
        
    # 计算 FPS
    fps_frame_count += 1
    if fps_frame_count >= 10:  # 每10帧计算一次FPS
        fps_end_time = time.time()
        fps = fps_frame_count / (fps_end_time - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()
    
    # 在左上角显示 FPS
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("camera", display_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # q to quit
        print('Quit')
        break
        
cv2.destroyAllWindows()
cap.release()