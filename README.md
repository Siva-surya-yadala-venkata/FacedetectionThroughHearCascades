## 📸 Face Detection and Recognition System

This repository contains two Python scripts for performing **face detection, data collection, and recognition** using the **OpenCV** library with the **LBPH Face Recognizer**.

- `MultipleFaceDetection.py`: Recognizes multiple faces in real-time.
- `SingleFaceDetection.py`: Verifies if a **specific person** is present in the frame.

---

## 🔧 Features

### ✅ Common Features
- Face detection using Haar cascades.
- Face data collection and augmentation.
- Model training using LBPH face recognizer.
- CLAHE preprocessing for contrast enhancement.

### 🎯 MultipleFaceDetection.py
- Detects and recognizes **multiple known faces** in real-time.
- Saves 300 face images per person for training.
- Supports data augmentation to improve model robustness.
- Displays the name and confidence score on the video stream.

### 👤 SingleFaceDetection.py
- Detects whether a **specific person** is present.
- Saves 100 images per person.
- Asks for the person's name before recognition and verifies only against that.

---

## 📁 Directory Structure

```
.
├── MultipleFaceDetection.py
├── SingleFaceDetection.py
├── face_dataset/
│   └── person_name/
│       ├── 0.jpg
│       └── ...
├── face_model.yml
└── label_map.txt
```

---

## ⚙️ Requirements

- Python 3.x
- OpenCV (with `contrib` for `face` module)
- NumPy

### 💡 Install with pip:
```bash
pip install opencv-contrib-python numpy
```

---

## 🚀 How to Run

### 1. **Collect Face Data**
Run either script and choose option 1:

```bash
python MultipleFaceDetection.py
# OR
python SingleFaceDetection.py
```

Enter the name of the person when prompted and press `q` to stop capturing early if needed.

---

### 2. **Train the Model**
Choose option 2 in the menu:

```bash
2. Train Model
```

This will save:
- Trained model: `face_model.yml`
- Label mapping: `label_map.txt`

---

### 3. **Recognize Faces**
Choose option 3:

- For **Multiple**: All trained faces will be identified.
- For **Single**: Will check if the specific person you entered is recognized.

Press `q` to exit the webcam.

---

## 📌 Notes

- For best results, capture face images in **good lighting** and **neutral expressions**.
- Images are resized to 200x200 pixels.
- You can tune `minNeighbors`, `scaleFactor`, and CLAHE parameters as needed.
- Thresholds are configurable for recognition accuracy.

---

## 🙌 Author

Made by **Yadala Venkata Siva Surya**  
Feel free to contribute or raise issues if you'd like to improve the project!
