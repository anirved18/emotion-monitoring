import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLO Face Detector
face_model = YOLO("models/yolov10n-face.pt")

# Load Emotion Model
emotion_model = load_model("models/emotion_model_finetuned.h5")

# Emotion Labels
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Temporal Smoothing Buffer
emotion_buffer = []
buffer_size = 5

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    results = face_model(frame, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy

        for box in boxes:
            x1,y1,x2,y2 = map(int,box)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face,(224,224))
            face = face/255.0
            face = np.reshape(face,(1,224,224,3))

            pred = emotion_model.predict(face, verbose=0)
            emotion_buffer.append(pred)

            if len(emotion_buffer) > buffer_size:
                emotion_buffer.pop(0)

            avg_pred = np.mean(emotion_buffer, axis=0)

            emotion = emotions[np.argmax(avg_pred)]
            conf = np.max(avg_pred)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            text = emotion + " " + str(round(conf*100,2)) + "%"
            cv2.putText(frame,text,(x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Emotion Monitoring",frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()