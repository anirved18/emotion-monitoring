import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import gdown
import os

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("Real-Time Emotion Detection")

MODEL_PATH = "emotion_model_finetuned.h5"

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1GkFhhdDBo_8ANdZ6XRu6YFUIOH0RkCm0"
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

model = load_emotion_model()

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.current_emotion = ""

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # 🔥 Smaller frame for faster detection
        small = cv2.resize(img, (320,240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )

        self.frame_count += 1

        for (x, y, w, h) in faces:

            # 🔥 Scale back to original frame
            x = int(x * 2)
            y = int(y * 2)
            w = int(w * 2)
            h = int(h * 2)

            if self.frame_count % 10 == 0:

                face = img[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face,(224,224))
                face = face.astype("float32")/255.0
                face = np.expand_dims(face, axis=0)

                prediction = model.predict(face, verbose=0)
                self.current_emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,self.current_emotion,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False}
)