import cv2
from ultralytics import YOLO
import math

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the YOLO model (Nano version for speed)
        self.model = YOLO(model_path)
        
        # Classes we care about (COCO dataset indices)
        # 67: Cell phone, 73: Book, 77: Teddy bear (often mistaken for person), etc.
        # You can check standard COCO classes for others.
        self.target_classes = [67, 73] 

    def detect(self, frame, conf_threshold=0.30):
        """
        Detects objects in the frame.
        conf_threshold: 0.30 means we only need 30% certainty. 
                        Lower this if it still misses the phone.
        """
        results = self.model(frame, stream=True, verbose=False, conf=conf_threshold)
        
        is_threat = False
        threat_label = ""
        boxes = []

        for r in results:
            for box in r.boxes:
                # Get Class ID and Confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in self.target_classes:
                    is_threat = True
                    
                    # Get Box Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Label map
                    label_name = self.model.names[cls_id]
                    threat_label = f"{label_name} ({int(conf*100)}%)"
                    
                    boxes.append((x1, y1, x2, y2, threat_label))

        return is_threat, threat_label, boxes