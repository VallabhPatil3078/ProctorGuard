import cv2
from src.object_detector import ObjectDetector

def main():
    cap = cv2.VideoCapture(0)
    # Initialize (This will download yolov8n.pt ~6MB)
    detector = ObjectDetector() 
    
    print("Starting YOLO Test... Hold up a phone!")

    while True:
        success, frame = cap.read()
        if not success: break

        is_threat, label, boxes = detector.detect(frame)

        if is_threat:
            # Draw Red Box around the phone
            for (x1, y1, x2, y2, lbl) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("YOLO Test", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()