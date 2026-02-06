import streamlit as st
import cv2
import time
import math
import numpy as np
import collections
from src.face_detector import FaceDetector
from src.head_pose import HeadPoseEstimator
from src.gaze_estimator import GazeEstimator
from src.fusion_engine import FusionEngine
from src.logger import EvidenceLogger
from src.object_detector import ObjectDetector
from src.audio_monitor import AudioMonitor

# --- PAGE CONFIG ---
st.set_page_config(page_title="ProctorGuard Dashboard", layout="wide")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🔧 Proctor Settings")
st.sidebar.markdown("Adjust thresholds in real-time.")

head_yaw_limit = st.sidebar.slider("Head Turn Limit (Yaw)", 10, 60, 25)
head_pitch_limit = st.sidebar.slider("Head Tilt Limit (Pitch)", 10, 60, 25)
gaze_limit = st.sidebar.slider("Eye Gaze Limit", 10, 60, 20)
time_limit = st.sidebar.slider("Suspicion Timer (Sec)", 0.5, 5.0, 1.0)

# --- OPTIMIZATION SETTINGS ---
FRAME_SKIP_FACE = 2      # Reduced skip to 2 for smoother eye tracking
FRAME_SKIP_OBJECT = 10   # Check objects more often (every 10 frames) to catch "quick" movements
RESIZE_WIDTH = 720       # Increased resolution for better eye detail

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_models():
    detector = FaceDetector()
    head_pose = HeadPoseEstimator()
    gaze_estimator = GazeEstimator("models/L2CSNet_gaze360.pkl")
    object_detector = ObjectDetector() # Uses updated confidence inside class
    audio_monitor = AudioMonitor(threshold=7)
    return detector, head_pose, gaze_estimator, object_detector, audio_monitor

# Initialize Session State
if 'fusion_engine' not in st.session_state:
    st.session_state.fusion_engine = FusionEngine()
    st.session_state.logger = EvidenceLogger()
    (st.session_state.detector, 
     st.session_state.head_pose, 
     st.session_state.gaze_estimator, 
     st.session_state.object_detector, 
     st.session_state.audio_monitor) = load_models()

# Update Engine dynamically
st.session_state.fusion_engine.HEAD_YAW_THRESH = head_yaw_limit
st.session_state.fusion_engine.HEAD_PITCH_THRESH = head_pitch_limit
st.session_state.fusion_engine.GAZE_THRESH = gaze_limit
st.session_state.fusion_engine.TIME_THRESH = time_limit

# --- HELPER: 3D Vector Drawing ---
def draw_vector(frame, pitch, yaw, origin_x, origin_y, length=100, color=(255, 0, 0)):
    if pitch is None or yaw is None: return
    pitch_rad = pitch * np.pi / 180
    yaw_rad = yaw * np.pi / 180
    end_x = int(origin_x + length * np.sin(yaw_rad))
    end_y = int(origin_y - length * np.sin(pitch_rad))
    cv2.arrowedLine(frame, (origin_x, origin_y), (end_x, end_y), color, 3)

# --- APP LOGIC ---
st.title("🛡️ ProctorGuard: AI Proctoring System")

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Live Proctoring Feed")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()
with col2:
    st.subheader("Real-Time Metrics")
    metric_status = st.empty()

if st.button("Start Exam Session"):
    detector = st.session_state.detector
    head_pose = st.session_state.head_pose
    gaze_estimator = st.session_state.gaze_estimator
    object_detector = st.session_state.object_detector
    audio_monitor = st.session_state.audio_monitor

    cap = cv2.VideoCapture(0)
    
    # Calibration & Timing
    calib_frames = []
    calib_start_time = time.time()
    is_calibrated = False
    stop_button = st.button("Stop Exam")
    audio_monitor.start()
    frame_count = 0
    
    # SMOOTHING BUFFER (The Fix for Jittery Eyes)
    # Stores last 5 frames of gaze data to smooth out noise
    gaze_buffer_pitch = collections.deque(maxlen=5)
    gaze_buffer_yaw = collections.deque(maxlen=5)

    # Persistence Variables
    current_landmarks = None
    current_h_pitch, current_h_yaw = 0, 0
    current_g_pitch, current_g_yaw = 0, 0
    current_nose = (0, 0)
    current_boxes = []
    current_threat_label = ""
    is_threat = False
    is_talking = False
    vol_level = 0
    is_cheating = False
    reason = "Safe"
    status_color = (0, 255, 0)

    while cap.isOpened() and not stop_button:
        success, frame = cap.read()
        if not success: break
        
        # 1. Resize & Flip
        frame = cv2.flip(frame, 1)
        h_orig, w_orig = frame.shape[:2]
        if w_orig > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w_orig
            dim = (RESIZE_WIDTH, int(h_orig * scale))
            frame = cv2.resize(frame, dim)
        h, w, _ = frame.shape
        frame_count += 1
        elapsed_time = time.time() - calib_start_time

        # --- PHASE 1: CALIBRATION ---
        if not is_calibrated:
            # A. Check for Objects (Security Patch)
            if frame_count % FRAME_SKIP_OBJECT == 0:
                is_threat, current_threat_label, current_boxes = object_detector.detect(frame, conf_threshold=0.30)
                if is_threat:
                    st.toast(f"Illegal Object: {current_threat_label}. Restarting...", icon="🚫")
                    calib_start_time = time.time() # Reset Timer
                    calib_frames = []

            # B. Check Face (Smoothing Enabled)
            if frame_count % FRAME_SKIP_FACE == 0:
                current_landmarks = detector.detect(frame)
                if current_landmarks:
                    hp, hy, _ = head_pose.estimate(frame, current_landmarks)
                    gp, gy = gaze_estimator.estimate(frame, current_landmarks)
                    
                    # Convert to degrees
                    gy_deg = gy * (180.0 / math.pi)
                    gp_deg = gp * (180.0 / math.pi)
                    
                    # Store data
                    if elapsed_time > 2 and not is_threat:
                        calib_frames.append([hp, hy, gp_deg, gy_deg])
                        
                    # Update current vars for display
                    current_h_pitch, current_h_yaw = hp, hy
                    current_g_pitch, current_g_yaw = gp_deg, gy_deg

            # UI
            remaining = 5 - int(elapsed_time)
            msg = f"CALIBRATING: {remaining}s" if not is_threat else "REMOVE PHONE!"
            color = (0, 255, 255) if not is_threat else (0, 0, 255)
            cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Validation
            if elapsed_time > 5 and not is_threat:
                if calib_frames:
                    avg = np.mean(calib_frames, axis=0)
                    # Relaxed validation check (30 degrees) to prevent frustration
                    if abs(avg[0]) > 30 or abs(avg[1]) > 30:
                        st.toast("Look straight at the screen!", icon="❌")
                        calib_frames = []
                        calib_start_time = time.time()
                    else:
                        st.session_state.fusion_engine.set_calibration(avg[0], avg[1], avg[2], avg[3])
                        is_calibrated = True
                        st.toast("Calibration Complete!", icon="✅")
                else:
                    calib_start_time = time.time()

        # --- PHASE 2: ACTIVE EXAM ---
        else:
            # A. Face & Gaze (Buffered Smoothing)
            if frame_count % FRAME_SKIP_FACE == 0:
                current_landmarks = detector.detect(frame)
                if current_landmarks:
                    # Raw Values
                    hp, hy, _ = head_pose.estimate(frame, current_landmarks)
                    gp, gy = gaze_estimator.estimate(frame, current_landmarks)
                    
                    # Add to Buffer
                    gaze_buffer_pitch.append(gp * (180.0 / math.pi))
                    gaze_buffer_yaw.append(gy * (180.0 / math.pi))
                    
                    # Calculate Smoothed Average
                    avg_gp = sum(gaze_buffer_pitch) / len(gaze_buffer_pitch)
                    avg_gy = sum(gaze_buffer_yaw) / len(gaze_buffer_yaw)
                    
                    current_h_pitch, current_h_yaw = hp, hy
                    current_g_pitch, current_g_yaw = avg_gp, avg_gy
                    
                    nose = current_landmarks.landmark[1]
                    current_nose = (int(nose.x * w), int(nose.y * h))

            # B. Object Detection
            if frame_count % FRAME_SKIP_OBJECT == 0:
                is_threat, current_threat_label, current_boxes = object_detector.detect(frame, conf_threshold=0.30)

            # C. Audio Check
            is_talking, vol_level = audio_monitor.get_status()

            # --- DECISION LOGIC ---
            if is_threat:
                is_cheating = True
                reason = current_threat_label
                status_color = (0, 0, 255)
            elif is_talking:
                is_cheating = True
                reason = f"Audio: {vol_level}"
                status_color = (0, 0, 255)
            elif current_landmarks:
                # Use Smoothed Gaze Data
                is_cheating, reason = st.session_state.fusion_engine.analyze(
                    current_h_pitch, current_h_yaw, current_g_pitch, current_g_yaw
                )
                status_color = (0, 0, 255) if is_cheating else (0, 255, 0)
            else:
                is_cheating = False
                reason = "No Face"
                status_color = (255, 255, 0)

            # --- VISUALIZATION ---
            if current_landmarks:
                # Draw Gaze Vector
                draw_vector(frame, current_h_pitch, current_h_yaw, current_nose[0], current_nose[1])
                # Draw Debug Data (Helps you understand why it's detecting)
                debug_text = f"Gaze Y:{int(current_g_yaw)} P:{int(current_g_pitch)}"
                cv2.putText(frame, debug_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            if is_threat:
                for (x1, y1, x2, y2, lbl) in current_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if is_cheating:
                cv2.rectangle(frame, (0, 0), (w, h), status_color, 10)
                alert_placeholder.error(f"⚠️ {reason}")
            else:
                alert_placeholder.success("User Status: Safe")

            metric_status.metric("Gaze Yaw", f"{int(current_g_yaw)}°")
            st.sidebar.metric("🎤 Audio Level", vol_level)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    audio_monitor.stop()