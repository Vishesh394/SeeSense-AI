import time
from pathlib import Path

import cv2
import numpy as np

from modules.voice import speak

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

YOLO_MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
SPEAK_INTERVAL_SECONDS = 3

SSD_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "table",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]

_detector = None


class YoloDetector:
    name = "YOLOv8"

    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO(YOLO_MODEL_NAME)

    def detect(self, frame):
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        objects = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                start_x, start_y, end_x, end_y = box.xyxy[0].cpu().numpy().astype("int")

                objects.append({
                    "label": label,
                    "confidence": confidence,
                    "box": (
                        max(0, start_x),
                        max(0, start_y),
                        min(w - 1, end_x),
                        min(h - 1, end_y),
                    ),
                })

        return objects


class MobileNetSsdDetector:
    name = "MobileNetSSD"

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            str(MODEL_DIR / "MobileNetSSD_deploy.prototxt"),
            str(MODEL_DIR / "MobileNetSSD_deploy.caffemodel"),
        )

    def detect(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        objects = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            idx = int(detections[0, 0, i, 1])
            if idx <= 0 or idx >= len(SSD_CLASSES):
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")

            objects.append({
                "label": SSD_CLASSES[idx],
                "confidence": confidence,
                "box": (
                    max(0, start_x),
                    max(0, start_y),
                    min(w - 1, end_x),
                    min(h - 1, end_y),
                ),
            })

        return objects


def get_detector():
    global _detector

    if _detector is not None:
        return _detector

    try:
        _detector = YoloDetector()
    except Exception as exc:
        print(f"[detection] YOLO is unavailable, using MobileNetSSD instead: {exc}")
        _detector = MobileNetSsdDetector()

    print(f"[detection] Active detector: {_detector.name}")
    return _detector


def detect_objects(frame):
    objects = get_detector().detect(frame)
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


def draw_objects(frame, objects):
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

        draw_objects(frame, objects)
        cv2.putText(
            frame,
            f"Detector: {get_detector().name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
