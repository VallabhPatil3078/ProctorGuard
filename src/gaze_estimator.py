import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from src.l2cs_model import get_l2cs_model
import torch.nn.functional as F

class GazeEstimator:
    def __init__(self, weights_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # 1. Load Model
        self.model = get_l2cs_model()
        
        # --- FIX: Added strict=False to ignore the unused 'fc_finetune' layer ---
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Define Preprocessing (Standard ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.softmax = torch.nn.Softmax(dim=1)
        self.idx_tensor = [i for i in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def estimate(self, frame, face_landmarks):
        # 1. Bounding Box Logic (Simple version based on landmarks)
        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
        ys = [int(lm.y * h) for lm in face_landmarks.landmark]
        
        x_min, x_max = max(0, min(xs)), min(w, max(xs))
        y_min, y_max = max(0, min(ys)), min(h, max(ys))
        
        # Expand box slightly to get full face
        face_img = frame[y_min:y_max, x_min:x_max]
        
        if face_img.size == 0:
            return 0.0, 0.0

        # 2. Preprocess
        img_tensor = self.transform(face_img).unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            yaw_predicted, pitch_predicted = self.model(img_tensor)
            
            # 4. Post-processing (Convert Softmax bins to Angle)
            # Yaw
            yaw_predicted = self.softmax(yaw_predicted)
            yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 4 - 180
            
            # Pitch
            pitch_predicted = self.softmax(pitch_predicted)
            pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 4 - 180
            
            # Convert to degrees (numpy)
            degree_yaw = yaw_predicted.item() * np.pi / 180.0
            degree_pitch = pitch_predicted.item() * np.pi / 180.0
            
        return degree_pitch, degree_yaw