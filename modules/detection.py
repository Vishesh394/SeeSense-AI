import cv2
import numpy as np

def start_camera():
    cap=cv2.VideoCapture(0)

    while True:
        ret,frame=cap.read()
        if not ret:
            break

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF== ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


#Load model
net=cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "table",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

def detect_objects(frame):
    h,w=frame.shapr[:2]

    blob=cv2.dnn.blobFronImage(frame, 0.007843,(300,300),127.5)
    net.setInput(blob)
    detections=net.forward()

    objects=[]

    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]

        if confidence> 0.5:
            idx=int(detections[0,0,i,1])
            label=CLASSES[idx]
            objects.append(label)

    return objects

