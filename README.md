# 🎭 Deepfake Video Detection using Deep Learning

An AI-powered Deepfake Video Detection System built using **TensorFlow, Keras, MobileNetV2, OpenCV, and CNN-based Deep Learning** to classify videos as **REAL** or **FAKE**.

---

## 🚀 Project Overview

This project detects manipulated or AI-generated fake videos by analyzing facial patterns from video frames using deep learning techniques.

The system:
- Extracts frames from videos 🎥
- Detects faces using MTCNN 🧠
- Applies preprocessing and augmentation ⚙️
- Uses MobileNetV2 for classification 🤖
- Predicts whether the video is REAL or FAKE ✅❌

---

## 🛠️ Tech Stack

- 🐍 Python
- 🔥 TensorFlow
- 🧠 Keras
- 👁️ OpenCV
- 📸 MTCNN
- 🤖 MobileNetV2
- 📊 CNN (Convolutional Neural Network)

---

## 📂 Project Structure

```bash
deepfake_detection/
│
├── dataset/
│   ├── real/
│   └── fake/
│
├── frames/
│   ├── real/
│   └── fake/
│
├── frame_extraction.py
├── data_preprocessing.py
├── model.py
├── train_model.py
├── predict_video.py
├── requirements.txt
└── README.md
```

---

## ⚡ Features

✅ Deepfake video classification  
✅ Face detection using MTCNN  
✅ Transfer Learning with MobileNetV2  
✅ CNN-based prediction system  
✅ Video frame extraction  
✅ Real/Fake prediction pipeline  
✅ Optimized inference using frame skipping  

---

## 📊 Model Performance

| Metric | Result |
|--------|--------|
| Validation Accuracy | ~72% |
| Deep Learning Model | MobileNetV2 |
| Dataset | Celeb-DF |

---

## 📦 Dataset

Dataset used:

📌 Celeb-DF Dataset

https://www.kaggle.com/datasets/reubensuju/celeb-df-v2/data

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/deepfake-video-detection.git
cd deepfake-video-detection
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1️⃣ Extract Frames

```bash
python frame_extraction.py
```

---

### 2️⃣ Train Model

```bash
python train_model.py
```

---

### 3️⃣ Predict Video

```bash
python predict_video.py
```

---

## 🧠 Deep Learning Workflow

```text
Video Input
     ↓
Frame Extraction
     ↓
Face Detection
     ↓
Image Preprocessing
     ↓
MobileNetV2 CNN Model
     ↓
REAL / FAKE Prediction
```

---

## 🎯 Future Improvements

- 🔥 Improve model accuracy
- ⚡ Real-time webcam detection
- 🌐 Streamlit web application
- 🧠 Advanced architectures (EfficientNet/XceptionNet)
- 📈 Better temporal analysis

---

## 🎓 Learning Outcomes

- Deep Learning
- Computer Vision
- CNN Architecture
- Transfer Learning
- Face Detection
- Video Processing
- TensorFlow & Keras

---

## 👨‍💻 Author

**Akash Kadam**

Computer Science Engineering Student  
Passionate about AI, Machine Learning, and Deep Learning 🚀

---

## ⭐ If You Like This Project

Give this repository a ⭐ on GitHub!
