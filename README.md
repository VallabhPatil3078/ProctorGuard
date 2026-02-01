# ProctorGuard: Multi-Modal AI Proctoring System

## 📌 Abstract
ProctorGuard is a robust, non-intrusive automated proctoring system designed for remote examinations. It goes beyond traditional eye-tracking by utilizing a **Multi-Modal Fusion Engine** that combines **Computer Vision**, **Deep Learning**, and **Audio Forensics**. 

The system integrates **Geometric Head Pose (PnP)**, **Gaze Estimation (L2CS-Net)**, **Object Detection (YOLOv8)**, and **Real-Time Audio Analysis** to distinguish between natural student behaviors and genuine cheating attempts like using a phone, talking to someone, or looking away.

## 🚀 Key Features
- **Multi-Modal Cheat Detection:**
  - **Vision:** Tracks Head Pose (Pitch/Yaw) and Eye Gaze (L2CS-Net).
  - **Object:** Uses **YOLOv8** to detect unauthorized objects (e.g., Mobile Phones) with motion-blur handling.
  - **Audio:** Monitors microphone input for high-amplitude speech or whispering.
- **Dynamic Calibration:** A 5-second startup phase learns the user's natural seating position to prevent false positives.
- **Smart Priority Logic:** The Fusion Engine prioritizes "Hard Evidence" (Phone/Audio) over "Soft Evidence" (Gaze) to reduce false alarms.
- **Evidence Logger:** Automatically captures timestamped screenshots with the specific violation reason burned into the image.
- **Examiner Dashboard:** Professional **Streamlit** web interface for real-time monitoring and sensitivity adjustment.

## 🛠️ Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt