import cv2
import os
import time
from datetime import datetime
from threading import Thread

class EvidenceLogger:
    def __init__(self, folder="logs"):
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.last_log_time = 0
        self.cooldown = 2.0  # Only save 1 image every 2 seconds

    def log(self, frame, reason):
        current_time = time.time()
        
        # Prevent spamming thousands of images
        if current_time - self.last_log_time < self.cooldown:
            return

        # Run file saving in a separate thread to prevent lag
        t = Thread(target=self._save_image, args=(frame.copy(), reason))
        t.start()
        self.last_log_time = current_time

    def _save_image(self, img, reason):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.folder, f"CHEAT_{timestamp}.jpg")
        
        # Burn the reason into the image
        cv2.putText(img, f"ALERT: {reason}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, timestamp, (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        cv2.imwrite(filename, img)
        print(f"[Evidence Saved] {filename}")