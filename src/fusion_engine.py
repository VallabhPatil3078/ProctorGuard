import time

class FusionEngine:
    def __init__(self, head_yaw_thresh=30, head_pitch_thresh=35, gaze_thresh=30, time_thresh_sec=1.5):
        self.HEAD_YAW_THRESH = head_yaw_thresh
        self.HEAD_PITCH_THRESH = head_pitch_thresh
        self.GAZE_THRESH = gaze_thresh
        self.TIME_THRESH = time_thresh_sec
        
        self.suspicious_start_time = None
        
        # Calibration Offsets (Will be set during startup)
        self.offset_head_yaw = 0
        self.offset_head_pitch = 0
        self.offset_gaze_yaw = 0
        self.offset_gaze_pitch = 0

    def set_calibration(self, h_p, h_y, g_p, g_y):
        self.offset_head_pitch = h_p
        self.offset_head_yaw = h_y
        self.offset_gaze_pitch = g_p
        self.offset_gaze_yaw = g_y
        print(f"Calibration Set! Offsets -> Head: {h_p:.1f}/{h_y:.1f}, Gaze: {g_p:.1f}/{g_y:.1f}")

    def analyze(self, head_pitch, head_yaw, gaze_pitch, gaze_yaw):
        # 1. Apply Calibration (Zero out the user's natural position)
        rel_h_p = head_pitch - self.offset_head_pitch
        rel_h_y = head_yaw - self.offset_head_yaw
        rel_g_p = gaze_pitch - self.offset_gaze_pitch
        rel_g_y = gaze_yaw - self.offset_gaze_yaw

        is_suspicious = False
        reason = ""

        # 2. Priority Logic (Check Yaw first as it's most obvious)
        # Head Yaw
        if abs(rel_h_y) > self.HEAD_YAW_THRESH:
            is_suspicious = True
            reason = f"Head Turned Side ({int(rel_h_y)})"
        
        # Head Pitch (Up/Down)
        elif abs(rel_h_p) > self.HEAD_PITCH_THRESH:
            is_suspicious = True
            reason = f"Head Tilted Up/Down ({int(rel_h_p)})"

        # Gaze Yaw (Eyes Side) - Only if head is safe
        elif abs(rel_g_y) > self.GAZE_THRESH:
            is_suspicious = True
            reason = f"Eyes Looking Side ({int(rel_g_y)})"

        # Gaze Pitch (Eyes Up/Down)
        elif abs(rel_g_p) > self.GAZE_THRESH:
            is_suspicious = True
            reason = f"Eyes Looking Up/Down ({int(rel_g_p)})"
            
        # 3. Temporal Filter
        if is_suspicious:
            if self.suspicious_start_time is None:
                self.suspicious_start_time = time.time()
            
            elapsed = time.time() - self.suspicious_start_time
            if elapsed > self.TIME_THRESH:
                return True, reason
        else:
            self.suspicious_start_time = None
            
        return False, ""