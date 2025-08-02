import cv2
import os

# تحميل النموذج المدرب
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trained.yml")

# تحميل الكلاسيفاير لكشف الوجوه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ربط المعرفات بالأسماء
label_map = {
    0: "Batman",
    1: "Superman"
}

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)

        if conf < 70:  # كل ما كانت أقل كل ما كان أدق
            name = label_map.get(id_, "Unknown")
        else:
            name = "Unknown"

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()