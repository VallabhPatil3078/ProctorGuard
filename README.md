# ProctorGuard: Fusion-Based Online Proctoring System

## 📌 Abstract
ProctorGuard is a non-intrusive, automated proctoring system designed for remote examinations. Unlike traditional systems that rely solely on head pose, ProctorGuard utilizes a **Multi-Modal Fusion Engine** that combines **Geometric Head Pose Estimation (PnP)** and **Deep Learning Gaze Estimation (L2CS-Net)**. This allows it to distinguish between natural movements and "Head-Forward, Eyes-Away" cheating behaviors.

## 🚀 Key Features
- **Deep Learning Gaze Tracking:** Uses ResNet-50 (L2CS-Net) for precise pupil tracking.
- **Dynamic Calibration:** A 5-second startup phase learns the user's natural seating position to prevent false positives.
- **Smart Fusion Logic:** Prioritizes flagrant actions (Head Turn) over subtle ones (Eye Glance) with temporal filtering.
- **Evidence Logger:** Automatically captures and timestamps screenshots of suspicious activity.
- **Examiner Dashboard:** Real-time Streamlit web interface for monitoring.

## 🛠️ Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt