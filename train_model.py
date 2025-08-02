import cv2
import os
import numpy as np
from PIL import Image # type: ignore

# مسار مجلد الصور
dataset_path = 'dataset'

# تحميل الوجوه والـ labels
faces = []
labels = []
label_map = {}
current_label = 0

# قراءة الصور من المجلد
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue
    label_map[current_label] = person_name
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = Image.open(image_path).convert('L')  # تحويل لصورة رمادية
        img_np = np.array(img, 'uint8')
        faces.append(img_np)
        labels.append(current_label)
    current_label += 1

# التحقق من وجود بيانات
if len(faces) == 0:
    print("❌ لا توجد صور تدريب.")
    exit()

# إنشاء وتدريب النموذج
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# حفظ النموذج
recognizer.save("trainer.yml")

# حفظ أسماء الأشخاص
with open("labels.txt", "w") as f:
    for label, name in label_map.items():
        f.write(f"{label}:{name}\n")

print("✅ تم تدريب النموذج بنجاح وحفظه في trainer.yml")
