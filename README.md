# Face Detection & Recognition using PCA (Eigenfaces)

A complete Computer Vision project for **Face Detection**, **Face Recognition using PCA (Eigenfaces)**, and **Performance Evaluation using ROC Curve & Accuracy Metrics** implemented using **C++**, **OpenCV**, and **Qt**.

---

# 📌 Project Overview

This project is divided into three major modules:

1. **Face Detection**
   - Detects human faces from images using Haar Cascade Classifier.

2. **Face Recognition using PCA (Eigenfaces)**
   - Recognizes and classifies faces using Principal Component Analysis (PCA).

3. **Performance Evaluation**
   - Evaluates recognition performance using:
     - ROC Curve
     - Accuracy
     - Threshold Analysis

---

# 🧠 Concepts Used

## Face Detection
- Haar Cascade Classifier
- OpenCV Object Detection
- Grayscale Conversion
- Bounding Box Extraction

## Face Recognition
- Principal Component Analysis (PCA)
- Eigenfaces
- Dimensionality Reduction
- Euclidean Distance Classification

## Evaluation
- ROC Curve
- True Positive Rate (TPR)
- False Positive Rate (FPR)
- Accuracy Measurement

---

# 🖥️ GUI Features

The application provides an interactive Qt-based GUI containing:

- Image Loading
- Face Detection Visualization
- Face Recognition Results
- PCA Training & Testing
- ROC Curve Visualization
- Accuracy Display

---

# 📂 Project Structure

```text
Face-Recognition-Project/
│
├── data/
│   ├── train/
│   └── test/
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── src/
│   ├── main.cpp
│   ├── mainwindow.cpp
│   ├── mainwindow.h
│   ├── facedetection.cpp
│   ├── facepca.cpp
│   ├── roc.cpp
│   └── utils.cpp
│
├── ui/
│   └── mainwindow.ui
│
├── results/
│   ├── detection_results/
│   ├── recognition_results/
│   └── roc_curve/
│
├── README.md
└── CMakeLists.txt
```

# ⚙️ Technologies Used
- C++
- OpenCV
- Qt
- PCA (Eigenfaces)
- Haar Cascade
- CMake

# 🚀 How It Works
## 1️⃣ Face Detection

The system first:

- Loads an input image
- Converts it to grayscale
- Applies Haar Cascade face detection
- Draws bounding boxes around detected faces
- Detection Pipeline
```text
Input Image
     ↓
Grayscale Conversion
     ↓
Haar Cascade Detection
     ↓
Detected Faces
```

## 2️⃣ Face Recognition using PCA

After detecting faces:

- Faces are resized and normalized
- PCA extracts the most important facial features
- Faces are projected into Eigenface space
- Euclidean distance is used for classification
- Recognition Pipeline
```text
Detected Face
      ↓
Preprocessing
      ↓
PCA Projection
      ↓
Eigenface Representation
      ↓
Distance Comparison
      ↓
Identity Prediction
```

# 📊 ROC Curve & Accuracy

The project evaluates recognition quality using:

- True Positive Rate (TPR)
- False Positive Rate (FPR)
- Recognition Accuracy

The ROC Curve helps visualize:

- Classification quality
- Threshold sensitivity
- Recognition robustness

# 🧮 PCA (Eigenfaces) Overview

PCA reduces high-dimensional facial images into a lower-dimensional feature space.

The Eigenfaces approach:

- Computes principal components
- Keeps the most significant facial variations
- Reduces computation time
- Improves recognition efficiency

# ▶️ Build & Run
## Prerequisites

Install:
- OpenCV
- Qt Creator
- CMake
- MinGW / MSVC Compiler

## Build Instructions
## Using Qt Creator
1. Open the .pro or CMakeLists.txt
2. Configure the kit
3. Build the project
4. Run the application

## Using CMake
```bash
- mkdir build
- cd build
- cmake ..
make
```
# ▶️ Running the Application
1. Launch the GUI
2. Load an image
3. Run Face Detection
4. Train PCA model
5. Test Face Recognition
6. Evaluate ROC Curve

# 📷 Example Results
## Face Detection
- Accurate frontal face localization
- Bounding box visualization

## Face Recognition
- Identity prediction using Eigenfaces
- Distance-based matching

## ROC Analysis
- Performance visualization
- Accuracy vs Threshold analysis

# 🔍 Key Algorithms

| Module           | Algorithm          |
| ---------------- | ------------------ |
| Face Detection   | Haar Cascade       |
| Face Recognition | PCA / Eigenfaces   |
| Classification   | Euclidean Distance |
| Evaluation       | ROC Curve          |

# 📈 Advantages
- Fast face detection
- Efficient dimensionality reduction
- Lightweight recognition pipeline
- Interactive GUI
- Modular architecture

# ⚠️ Limitations
- Works best with frontal faces
- Sensitive to illumination changes
- PCA may struggle with large pose variations
- Haar Cascade may fail on occluded faces

# 🔮 Future Improvements
- Deep Learning Face Recognition
- CNN-based Detection
- LBPH / Fisherfaces Comparison
- Real-time Webcam Support
- Dataset Augmentation
- GPU Acceleration

# 👨‍💻 Authors

Developed as a Computer Vision project for:

- Face Detection
- Face Recognition
- PCA & Eigenfaces
- Performance Analysis

# 📚 References
- OpenCV Documentation
- Eigenfaces Research Paper
- PCA Theory
- Viola-Jones Face Detection Framework

# ⭐ Acknowledgment

Special thanks to the Computer Vision and Image Processing community and OpenCV contributors for the foundational tools and algorithms used in this project.