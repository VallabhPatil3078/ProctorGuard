import cv2
import time
import math
import numpy as np
from src.face_detector import FaceDetector
from src.head_pose import HeadPoseEstimator
from src.gaze_estimator import GazeEstimator
from src.fusion_engine import FusionEngine
from src.logger import EvidenceLogger

# --- Helper: Draw 3D Projection Line ---
def draw_vector(frame, pitch, yaw, origin_x, origin_y, length=100, color=(255, 0, 0)):
    # Convert degrees to radians
    pitch_rad = pitch * np.pi / 180
    yaw_rad = yaw * np.pi / 180
    
    # Calculate end point
    end_x = int(origin_x + length * np.sin(yaw_rad))
    end_y = int(origin_y - length * np.sin(pitch_rad))
    
    cv2.arrowedLine(frame, (origin_x, origin_y), (end_x, end_y), color, 3)

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    head_pose = HeadPoseEstimator()
    gaze_estimator = GazeEstimator("models/L2CSNet_gaze360.pkl")
    fusion_engine = FusionEngine()
    logger = EvidenceLogger()
    
    # --- CALIBRATION STATE VARIABLES ---
    is_calibrated = False
    calib_frames = []
    calib_start_time = None
    
    print("System Started. Prepare for Calibration...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        face_landmarks = detector.detect(frame)
        
        if face_landmarks:
            # 1. Get Raw Metrics
            h_pitch, h_yaw, h_roll = head_pose.estimate(frame, face_landmarks)
            g_pitch, g_yaw = gaze_estimator.estimate(frame, face_landmarks)
            g_yaw_deg = g_yaw * (180.0 / math.pi)
            g_pitch_deg = g_pitch * (180.0 / math.pi)

            # Get Nose Position for visualization
            nose = face_landmarks.landmark[1] # Tip of nose
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)

            # --------------------------------------
            # PHASE 1: CALIBRATION
            # --------------------------------------
            if not is_calibrated:
                if calib_start_time is None:
                    calib_start_time = time.time()
                
                elapsed = time.time() - calib_start_time
                remaining = 5 - int(elapsed)
                
                # UI Instructions
                cv2.rectangle(frame, (0, 0), (w, h), (50, 50, 50), -1) # Dim screen
                cv2.putText(frame, "CALIBRATION MODE", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, "Please look at the CENTER of the screen", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Starting in: {remaining}s", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # Collect Data for last 2 seconds
                if elapsed > 3: 
                    calib_frames.append([h_pitch, h_yaw, g_pitch_deg, g_yaw_deg])
                
                if elapsed > 5:
                    # Calculate Averages
                    avg = np.mean(calib_frames, axis=0)
                    fusion_engine.set_calibration(avg[0], avg[1], avg[2], avg[3])
                    is_calibrated = True
                    print("Calibration Complete!")

                cv2.imshow('ProctorGuard', frame)
                if cv2.waitKey(5) & 0xFF == 27: break
                continue

            # --------------------------------------
            # PHASE 2: MONITORING
            # --------------------------------------
            
            # Draw Gaze Vector (Visual Feedback)
            draw_vector(frame, h_pitch, h_yaw, nose_x, nose_y)

            # Analyze
            is_cheating, reason = fusion_engine.analyze(h_pitch, h_yaw, g_pitch_deg, g_yaw_deg)

            color = (0, 255, 0)
            status_text = "Status: SAFE"
            
            if is_cheating:
                color = (0, 0, 255)
                status_text = f"ALERT: {reason}"
                cv2.rectangle(frame, (0, 0), (w, h), color, 5)
                
                # Log Evidence
                logger.log(frame, reason)

            # Display Stats
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Head: {int(h_pitch)}, {int(h_yaw)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Gaze: {int(g_pitch_deg)}, {int(g_yaw_deg)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('ProctorGuard', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()