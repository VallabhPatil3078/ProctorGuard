import sounddevice as sd
import numpy as np

class AudioMonitor:
    def __init__(self, threshold=15.0):
        self.volume = 0
        self.threshold = threshold
        self.stream = None
        self.is_running = False

    def callback(self, indata, frames, time, status):
        # Calculate Volume (RMS Amplitude)
        # indata contains the raw audio samples
        if status:
            print(status)
        
        # Calculate the "Loudness" number
        volume_norm = np.linalg.norm(indata) * 10
        self.volume = int(volume_norm)

    def start(self):
        # Start listening in a non-blocking background thread
        if self.stream is None:
            self.stream = sd.InputStream(callback=self.callback)
            self.stream.start()
            self.is_running = True
            print("Audio Monitor Started.")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_running = False

    def get_status(self):
        # Return boolean (Is Talking?) and the raw volume number
        is_talking = self.volume > self.threshold
        return is_talking, self.volume