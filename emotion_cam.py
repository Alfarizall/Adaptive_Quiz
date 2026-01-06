import cv2
import numpy as np
from keras.models import load_model

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

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        prediction = emotion_model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        confidence = np.max(prediction)

        label = f"{emotion_labels[emotion_index]} ({confidence*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)
        cv2.putText(frame, label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Deteksi Ekspresi Wajah", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
