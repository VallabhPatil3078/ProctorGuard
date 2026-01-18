import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        # 1. Standard 3D Face Model (Generic)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype="double")

        # MediaPipe Landmark Indices
        self.key_landmarks = [1, 199, 33, 263, 61, 291]

    def estimate(self, image, face_landmarks):
        img_h, img_w, _ = image.shape

        # 2. Extract 2D coordinates
        image_points = []
        for idx in self.key_landmarks:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            image_points.append((x, y))
        
        image_points = np.array(image_points, dtype="double")

        # 3. Camera Internals
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        # 4. Solve PnP
        success, vector_rotation, vector_translation = cv2.solvePnP(
            self.model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # 5. Math: Rotation Vector -> Euler Angles
        rmat, _ = cv2.Rodrigues(vector_rotation)
        
        # Calculate Euler Angles from Rotation Matrix
        sy = np.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rmat[2,1] , rmat[2,2])
            y = np.arctan2(-rmat[2,0], sy)
            z = np.arctan2(rmat[1,0], rmat[0,0])
        else:
            x = np.arctan2(-rmat[1,2], rmat[1,1])
            y = np.arctan2(-rmat[2,0], sy)
            z = 0

        # Convert to Degrees
        pitch = x * 180.0 / np.pi
        yaw = y * 180.0 / np.pi
        roll = z * 180.0 / np.pi

        # -------------------------------------------------------------
        # FIX: THE "180" OFFSET CORRECTION
        # -------------------------------------------------------------
        # If pitch is reading ~180 (looking backwards), flip it to 0.
        # This usually happens because of PnP coordinate assumptions.
        
        if pitch > 0:
            pitch = 180 - pitch
        else:
            pitch = -180 - pitch
            
        # Often PnP yaw is also inverted in this frame
        # (Optional: Test this. If looking Left gives Positive, it's correct)
        
        return pitch, yaw, roll