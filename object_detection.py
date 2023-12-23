
# --------------------------------------------------
# File Name: object_detection.py
# --------------------------------------------------
# Date Completed: 08-27-2023
# --------------------------------------------------
# Description:
# This is a simple object detection program that 
# uses the YOLO (You Only Look Once) model to detect
# and identify objects in a real-time webcam feed.
# --------------------------------------------------

# Import Dependencies
from ultralytics import YOLO
import cv2
import math

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 520)

# YOLO Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object Classes
classNames = ["Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck", "Boat",
              "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird", "Cat",
              "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella",
              "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
              "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup",
              "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli",
              "Carrot", "Hot dog", "Pizza", "Donut", "Cake", "Chair", "Sofa", "Potted Plant", "Bed",
              "Dining Table", "Toilet", "TV Monitor", "Laptop", "Mouse", "Remote", "Keyboard", "Mobile Phone",
              "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
              "Teddy Bear", "Hair Dryer", "Toothbrush"
              ]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip camera horizontally
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.line(img, (x1, y1), (x1 + 15, y1), (0, 255, 0), 2)                      # Top line
            cv2.line(img, (x1, y1), (x1, y1 + 15), (0, 255, 0), 2)                      # Left line
            cv2.line(img, (x2, y1), (x2 - 15, y1), (0, 255, 0), 2)                      # Right line
            cv2.line(img, (x1, y2), (x1 + 15, y2), (0, 255, 0), 2)                      # Bottom line
            cv2.line(img, (x1, y2), (x1, y2 - 15), (0, 255, 0), 2)                      # Bottom left line
            cv2.line(img, (x2, y2), (x2 - 15, y2), (0, 255, 0), 2)                      # Bottom right line
            cv2.line(img, (x2, y1), (x2, y1 + 15), (0, 255, 0), 2)                      # Right top line
            cv2.line(img, (x2, y2), (x2, y2 - 15), (0, 255, 0), 2)                      # Right bottom line

            # Draw confidence and class name
            confidence = round(float(box.conf[0]) * 100, 2)
            class_index = int(box.cls[0])
            class_name = classNames[class_index]
            text = f"{class_name}: {confidence}%"
            org = (x1, y1 - 10)  # Place text slightly above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(img, text, org, font, font_scale, color, thickness)

    cv2.imshow("Object Detection", img)
    if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
