from ultralytics import YOLO
import cv2
import cvzone
import math
import winsound  # For beep sound on Windows

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
#cap = cv2.VideoCapture("../yolo-webcam/ppe-2-1.mp4")  # For Video  ../yolo-web/ppe.mp4

model = YOLO("best.pt")

classNames = ['Boots', 'Ear-protection', 'Glass', 'Glove', 'Helmet', 'Mask', 'Person', 'Vest']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    helmet_detected = False  # Flag to check if helmet is detected

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5 and currentClass == 'Helmet':
                helmet_detected = True  # Set flag if helmet is detected
                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=(0, 255, 0),
                                   colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if not helmet_detected:
        # Alert message if no helmet is detected
        cvzone.putTextRect(img, 'No Helmet Detected', (50, 50), scale=2, thickness=2, colorB=(0, 0, 255),
                           colorT=(255, 255, 255), colorR=(0, 0, 255), offset=10)
        winsound.Beep(1000, 500)  # Beep sound with 1000 Hz frequency for 500 milliseconds

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
