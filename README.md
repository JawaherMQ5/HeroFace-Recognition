# 🧠 Face Recognition with OpenCV

This project demonstrates how to build a simple Face Recognition System using Python and OpenCV. It uses a dataset of face images to train a recognizer and then identifies faces in real time using a webcam.

---

## 📁 Project Structure

face-recognition-project/
├── dataset/
│   ├── Batman/
│   │   ├── 1.jpg
│   │   └── ...
│   └── Superman/
│       ├── 1.jpg
│       └── ...
├── train_model.py
├── recognize.py
├── face_trained.yml
└── README.md

---

## ⚙️ How to Run

### 1. ✅ Install Requirements

Make sure Python 3.11+ is installed. Then, install OpenCV:
Bash->
pip uninstall opencv-python
pip install opencv-contrib-python

---

### 2. 🖼️ Prepare Dataset

Create a folder named dataset/ with subfolders for each person. Each subfolder must contain several images of that person.

Example:
dataset/
├── Batman/
│   ├── 1.jpg
│   └── 2.jpg
└── Superman/
    ├── 1.jpg
    └── 2.jpg

---

### 3. 🧠 Train the Model

Train the face recognizer using the dataset:
Bash->
python train_model.py

This generates the model file face_trained.yml.

---

### 4. 🎥 Run Face Recognition

Start face detection and recognition via webcam:
Bash ->
python recognize.py

The script will show the webcam feed and recognize known faces.

---

## 📂 Files Description

| File               | Description                                   |
|--------------------|-----------------------------------------------|
| train_model.py   | Script to train the face recognition model    |
| recognize.py     | Script to detect and recognize faces          |
| dataset/         | Folder with images of each person             |
| face_trained.yml | Trained model saved after running training    |

---

## ✨ Notes

- Make sure face images are clear and well-lit.
- Use consistent image size and angle for better accuracy.
- You can add more people by adding folders to dataset/ and re-running training.

---
