# Breast Tumor Classification Using Deep Learning on Ultrasound Images

This repository presents the implementation of a deep learning-based pipeline for classifying breast ultrasound images into **benign**, **malignant**, and **normal** categories. Developed as part of the M.Tech program in **Artificial Intelligence and Data Science** at **Amrita Vishwa Vidyapeetham, Faridabad**, this project aims to support early and accessible breast cancer detection, particularly in resource-constrained environments.

---

## 🔍 Project Overview

Breast cancer is the most commonly diagnosed cancer worldwide. Early detection is crucial, and ultrasound imaging is a cost-effective diagnostic tool. This project utilizes **transfer learning** with state-of-the-art Convolutional Neural Networks (CNNs) to automate the classification process of breast ultrasound images.

- **Dataset**: 780 images (Kaggle)
  - 437 Benign
  - 210 Malignant
  - 133 Normal
- **Techniques**: Transfer learning, data augmentation, class weighting
- **Frameworks**: TensorFlow 2.15, Keras 2.15
- **Platform**: Google Colab (Pro)

---

## 🎯 Objectives

- Develop an automated classification system for breast ultrasound images.
- Prioritize high sensitivity in detecting malignant tumors.
- Address small dataset challenges through augmentation and stratified sampling.
- Create a reproducible, low-cost diagnostic pipeline.
- Lay the groundwork for clinical deployment and integration in telemedicine platforms.

---

## 🧪 Methodology

### ➤ Data Preprocessing
- Images resized (224x224 for VGG16/ResNet50, 299x299 for InceptionV3)
- Normalization using ImageNet statistics
- Augmentation: rotation, flips, zoom

### ➤ Dataset Split
- Stratified split: 70% training, 15% validation, 15% test

### ➤ Model Architecture
- Pre-trained models: **VGG16**, **ResNet50**, **InceptionV3**
- Custom classification head added and fine-tuned
- Class weights applied to mitigate imbalance

### ➤ Evaluation Metrics
- Accuracy
- Sensitivity (Malignant-focused)
- Specificity
- F1 Score
- Confusion matrices

---

## 📈 Results

| Model                  | Accuracy (%) | Test Loss | Sensitivity (Malignant, %) | Training Time (s) |
|------------------------|--------------|-----------|-----------------------------|--------------------|
| VGG16                  | 64.96        | 0.9251    | 77.42                       | 5349               |
| ResNet50               | 74.36        | 0.7554    | 45.16                       | 1955               |
| InceptionV3 (AdamW)    | 75.21        | 0.7460    | 61.29                       | 3015               |
| **InceptionV3 (RMSprop)** | **73.48**    | **0.7859** | **61.29**                   | **3015**           |

> InceptionV3 with RMSprop optimizer demonstrated a strong trade-off between accuracy and sensitivity, making it suitable for medical diagnosis applications.

---


