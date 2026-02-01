import streamlit as st
import cv2
import time
import math
import numpy as np
import os
from src.face_detector import FaceDetector
from src.head_pose import HeadPoseEstimator
from src.gaze_estimator import GazeEstimator
from src.fusion_engine import FusionEngine
from src.logger import EvidenceLogger
from src.object_detector import ObjectDetector

# --- PAGE CONFIG ---
st.set_page_config(page_title="ProctorGuard Dashboard", layout="wide")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🔧 Proctor Settings")
st.sidebar.markdown("Adjust thresholds in real-time.")

# Interactive Sliders
head_yaw_limit = st.sidebar.slider("Head Turn Limit (Yaw)", 10, 60, 30)
head_pitch_limit = st.sidebar.slider("Head Tilt Limit (Pitch)", 10, 60, 35)
gaze_limit = st.sidebar.slider("Eye Gaze Limit", 10, 60, 30)
time_limit = st.sidebar.slider("Suspicion Timer (Sec)", 0.5, 5.0, 1.5)

# Initialize Engine with Slider Values
if 'fusion_engine' not in st.session_state:
    st.session_state.fusion_engine = FusionEngine()
    st.session_state.logger = EvidenceLogger()
    st.session_state.object_detector = ObjectDetector()

# Update Engine dynamically
st.session_state.fusion_engine.HEAD_YAW_THRESH = head_yaw_limit
st.session_state.fusion_engine.HEAD_PITCH_THRESH = head_pitch_limit
st.session_state.fusion_engine.GAZE_THRESH = gaze_limit
st.session_state.fusion_engine.TIME_THRESH = time_limit

# --- MAIN LAYOUT ---
st.title("🛡️ ProctorGuard: AI Proctoring System")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Proctoring Feed")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Metrics")
    chart_yaw = st.empty()
    chart_pitch = st.empty()
    metric_status = st.empty()

# --- HELPER: 3D Vector Drawing ---
def draw_vector(frame, pitch, yaw, origin_x, origin_y, length=100, color=(255, 0, 0)):
    pitch_rad = pitch * np.pi / 180
    yaw_rad = yaw * np.pi / 180
    end_x = int(origin_x + length * np.sin(yaw_rad))
    end_y = int(origin_y - length * np.sin(pitch_rad))
    cv2.arrowedLine(frame, (origin_x, origin_y), (end_x, end_y), color, 3)

# --- APP LOGIC ---
if st.button("Start Exam Session"):
    # Initialization
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    head_pose = HeadPoseEstimator()
    gaze_estimator = GazeEstimator("models/L2CSNet_gaze360.pkl")
    
    # Calibration Vars
    calib_frames = []
    calib_start_time = time.time()
    is_calibrated = False
    
    stop_button = st.button("Stop Exam")

    while cap.isOpened() and not stop_button:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        face_landmarks = detector.detect(frame)
        
        status_color = (0, 255, 0) # Green
        status_text = "SAFE"
        
        if face_landmarks:
            # 1. Estimate
            h_pitch, h_yaw, h_roll = head_pose.estimate(frame, face_landmarks)
            g_pitch, g_yaw = gaze_estimator.estimate(frame, face_landmarks)
            g_yaw_deg = g_yaw * (180.0 / math.pi)
            g_pitch_deg = g_pitch * (180.0 / math.pi)
            
            # Nose coords for drawing
            nose = face_landmarks.landmark[1]
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)

            # 2. Calibration Phase (First 5 seconds)
            elapsed = time.time() - calib_start_time
            if not is_calibrated:
                remaining = 5 - int(elapsed)
                cv2.rectangle(frame, (0, 0), (w, h), (50, 50, 50), -1)
                cv2.putText(frame, f"CALIBRATING: {remaining}s", (180, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                if elapsed > 3: # Collect data
                    calib_frames.append([h_pitch, h_yaw, g_pitch_deg, g_yaw_deg])
                
                if elapsed > 5:
                    avg = np.mean(calib_frames, axis=0)
                    st.session_state.fusion_engine.set_calibration(avg[0], avg[1], avg[2], avg[3])
                    is_calibrated = True
                    st.toast("Calibration Complete!", icon="✅")
            
            else:
                # 3. Active Monitoring Phase
                
                # --- LAYER 1: OBJECT DETECTION (YOLO) ---
                # Run this FIRST. If a phone is visible, it's an instant alert.
                is_threat, threat_label, boxes = st.session_state.object_detector.detect(frame)
                
                if is_threat:
                    is_cheating = True
                    reason = threat_label
                    # Draw boxes around the phone/object
                    for (x1, y1, x2, y2, lbl) in boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                else:
                    # --- LAYER 2: BEHAVIORAL ANALYSIS (FUSION) ---
                    # Only check gaze if no phone is detected
                    draw_vector(frame, h_pitch, h_yaw, nose_x, nose_y)
                    is_cheating, reason = st.session_state.fusion_engine.analyze(
                        h_pitch, h_yaw, g_pitch_deg, g_yaw_deg
                    )

                # --- ALERT SYSTEM ---
                if is_cheating:
                    status_color = (0, 0, 255) # Red
                    status_text = f"ALERT: {reason}"
                    cv2.rectangle(frame, (0, 0), (w, h), status_color, 10)
                    
                    # Log Evidence (Works for both Phones and Gaze)
                    st.session_state.logger.log(frame, reason)
                    alert_placeholder.error(f"⚠️ Cheating Detected: {reason}")
                else:
                    alert_placeholder.success("User Status: Safe")

                # Update Sidebar Metrics
                metric_status.metric("Head Yaw (Turn)", f"{int(h_yaw)}°", delta_color="off")
        
        # Display Video in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_column_width=True)
        
        # Small sleep to reduce CPU usage
        time.sleep(0.01)

    cap.release()