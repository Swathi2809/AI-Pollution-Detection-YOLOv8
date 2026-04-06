# 🚗💨 AI-Based Vehicle Pollution Detection Using YOLOv8

## 📌 Overview

This project is an AI-based system that detects vehicle pollution using computer vision. It uses YOLOv8 models to identify vehicles and detect smoke emissions, then classifies pollution levels.

---

## 🎯 Features

- 🚗 Vehicle Detection (car, bus, truck, motorcycle)
- 💨 Smoke Detection using custom-trained YOLOv8 model
- 📊 Pollution Level Classification:
  - Low Pollution
  - Medium Pollution
  - High Pollution
- 🖼 Output image with bounding boxes and labels
- ⚠ Pollution alert display

---

## 🧠 Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy

---

## ⚙️ How It Works

1. Input image is loaded
2. Vehicle detection is performed using pretrained YOLOv8
3. Smoke detection is performed using custom model
4. Overlap between vehicle and smoke is checked
5. Pollution level is calculated based on:
   - Smoke confidence
   - Smoke area
6. Output is displayed and saved

---

## 🚀 Installation

```bash
pip install ultralytics opencv-python numpy
```
