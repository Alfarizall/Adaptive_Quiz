import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

emotion_model = load_model("emotion_model.h5", compile=False)

emotion_labels = {
    0: "Marah",
    1: "Jijik",
    2: "Takut / Cemas",
    3: "Senang",
    4: "Sedih",
    5: "Terkejut",
    6: "Netral / Bosan"
}

emotion_to_value = {
    "Marah": 0,
    "Jijik": 1,
    "Takut / Cemas": 2,
    "Sedih": 3,
    "Netral / Bosan": 4,
    "Terkejut": 5,
    "Senang": 6
}

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture("video_rekaman.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

time_series = []
emotion_series = []

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # wajah utama
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        prediction = emotion_model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion_text = emotion_labels[emotion_index]

        time_sec = frame_count / fps
        time_series.append(time_sec)
        emotion_series.append(
            emotion_to_value.get(emotion_text, 0)
        )

        cv2.putText(frame, emotion_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)

    frame_count += 1
    cv2.imshow("Deteksi Ekspresi", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# =============================
# GRAFIK PERUBAHAN EMOSI
# =============================
plt.figure()
plt.plot(time_series, emotion_series)
plt.yticks(
    list(emotion_to_value.values()),
    list(emotion_to_value.keys())
)
plt.xlabel("Waktu (detik)")
plt.ylabel("Emosi")
plt.title("Perubahan Emosi Sepanjang Video")
plt.show()
