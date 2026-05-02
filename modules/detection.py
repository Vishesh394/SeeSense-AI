import time
from pathlib import Path

import cv2
import numpy as np

from modules.voice import speak

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

net = cv2.dnn.readNetFromCaffe(
    str(MODEL_DIR / "MobileNetSSD_deploy.prototxt"),
    str(MODEL_DIR / "MobileNetSSD_deploy.caffemodel"),
)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "table",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]

CONFIDENCE_THRESHOLD = 0.45
SPEAK_INTERVAL_SECONDS = 3


def detect_objects(frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    objects = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        idx = int(detections[0, 0, i, 1])
        if idx <= 0 or idx >= len(CLASSES):
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")

        objects.append({
            "label": CLASSES[idx],
            "confidence": confidence,
            "box": (
                max(0, start_x),
                max(0, start_y),
                min(w - 1, end_x),
                min(h - 1, end_y),
            ),
        })

    return sorted(objects, key=lambda item: item["confidence"], reverse=True)


def describe_objects(objects):
    labels = []

    for obj in objects:
        label = obj["label"]
        if label not in labels:
            labels.append(label)

    if not labels:
        return ""

    if len(labels) == 1:
        return labels[0]

    return ", ".join(labels[:3])


def start_camera():
    cap = cv2.VideoCapture(0)

    last_spoken = ""
    last_time = 0

    if not cap.isOpened():
        print("[camera] Could not open webcam.")
        speak("Camera not available")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[camera] Could not read frame.")
            break

        objects = detect_objects(frame)
        spoken_text = describe_objects(objects)

        if spoken_text and (
            spoken_text != last_spoken
            or time.time() - last_time > SPEAK_INTERVAL_SECONDS
        ):
            speak(spoken_text)
            last_spoken = spoken_text
            last_time = time.time()

        for obj in objects:
            start_x, start_y, end_x, end_y = obj["box"]
            label = f"{obj['label']}: {obj['confidence'] * 100:.1f}%"

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            y = start_y - 10 if start_y - 10 > 10 else start_y + 20
            cv2.putText(
                frame,
                label,
                (start_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
