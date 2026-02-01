from ultralytics import YOLO
import cv2
import math

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        # Load the "Nano" model (Smallest & Fastest)
        # It will auto-download 'yolov8n.pt' on first run
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # COCO Dataset Classes (YOLO default)
        # ID 67: Cell Phone, ID 73: Book, ID 77: Teddy Bear (Just kidding), etc.
        # We define what constitutes a "Threat"
        self.forbidden_classes = [67] # 67 is 'cell phone' in COCO dataset

    def detect(self, frame):
        # Run inference
        results = self.model(frame, stream=True, verbose=False)
        
        detections = []
        is_threat = False
        threat_label = ""

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 1. Confidence Check
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf < self.conf_threshold:
                    continue
                
                # 2. Class Check
                cls = int(box.cls[0])
                
                # Only care if it's a forbidden object (e.g., Cell Phone)
                if cls in self.forbidden_classes:
                    is_threat = True
                    threat_label = "Mobile Phone Detected"
                    
                    # Get Coordinates for drawing
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    detections.append((x1, y1, x2, y2, threat_label))

        return is_threat, threat_label, detections