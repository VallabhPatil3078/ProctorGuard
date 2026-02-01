import time
from src.audio_monitor import AudioMonitor

def main():
    # threshold=10 is usually sensitive enough for a laptop mic
    monitor = AudioMonitor(threshold=10)
    monitor.start()
    
    print("🎤 Monitoring Audio... Please speak into your mic.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            is_talking, volume = monitor.get_status()
            
            # Draw a simple bar graph in the terminal
            bar = "█" * volume
            status = "🔴 TALKING" if is_talking else "🟢 SILENCE"
            
            # \r overwrites the same line so it looks like an animation
            print(f"\r{status} [Vol: {volume:03d}] {bar}", end="")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()

if __name__ == "__main__":
    main()