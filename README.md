# SeeSense-AI
To assist visually impaired individuals in navigating complex environments through real-time object recognition and obstacle detection, enhancing their independence and safety.

## Object detection

The app tries to use YOLOv8 first for better and wider object detection. If YOLO is not installed yet, it automatically falls back to the included MobileNetSSD model.

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
python app.py
```

The first YOLO run may download `yolov8n.pt`. After that, it should detect many more object types than MobileNetSSD.
