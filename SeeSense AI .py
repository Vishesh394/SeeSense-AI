import streamlit as st
import requests
import json
import numpy as np
import cv2
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Title and description
st.title("SeeSense AI: Visual Assistant for the Visually Impaired")
st.markdown("This app uses Google's Gemini AI to identify objects and detect obstacles in real-time, assisting visually impaired users with navigation.")

# Gemini API key (replace with your actual API key)
API_KEY = "AIzaSyCv6jQeMeO128Eq6MWKi-vOZlgcndHIHAQ"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key=AIzaSyCv6jQeMeO128Eq6MWKi-vOZlgcndHIHAQ"


# Function to get AI response from Gemini
def get_gemini_prediction(image_bytes):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
    response = requests.post(API_URL, headers=headers, files=files)
    if response.status_code == 200:
        return response.json().get("predictions", [])
    else:
        st.error("Failed to fetch predictions from Gemini AI")
        return []

# Real-time video capture
st.markdown("## Real-time Object and Obstacle Detection")
run = st.checkbox("Start Camera")

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert frame to bytes for API request
        _, img_encoded = cv2.imencode(".jpg", frame)
        img_bytes = img_encoded.tobytes()

        # Get AI predictions
        predictions = get_gemini_prediction(img_bytes)

        # Display predictions and detect obstacles
        obstacle_detected = False
        if predictions:
            for pred in predictions:
                label = pred.get('label', 'Unknown')
                confidence = pred.get('confidence', 0.0)
                if 'obstacle' in label.lower() or 'wall' in label.lower() or 'barrier' in label.lower():
                    obstacle_detected = True
                    cv2.putText(frame, f"Obstacle Detected: {label} ({confidence:.2f})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if obstacle_detected:
            st.warning("⚠️ Obstacle detected! Please proceed with caution.")
            engine.say("Obstacle detected! Please proceed with caution.")
            engine.runAndWait()

        # Display video feed
        stframe.image(frame, channels="BGR")

    cap.release()
