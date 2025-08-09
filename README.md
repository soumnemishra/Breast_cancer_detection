# 🏥 Breast Tumor Classification Using Deep Learning on Ultrasound Images

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-2.15-red.svg" alt="Keras">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Accuracy-73.48%25-brightgreen.svg" alt="Accuracy">
</div>

<div align="center">
  <h3>🎯 Early Breast Cancer Detection Through AI-Powered Ultrasound Analysis</h3>
  <p><em>Leveraging transfer learning and deep neural networks to classify breast ultrasound images into benign, malignant, and normal categories</em></p>
  
  <h4>👨‍🎓 Academic Project by <strong>Soumen Mishra</strong></h4>
  <p><em>M.Tech in Artificial Intelligence and Data Science</em><br>
  <em>Amrita Vishwa Vidyapeetham, Faridabad</em></p>
</div>

---

## 🎓 Academic Context

### 📚 Project Details
- **Student**: Soumen Mishra (DL.SC.P2AID24004)
- **Program**: M.Tech in Artificial Intelligence and Data Science
- **Institution**: Amrita Vishwa Vidyapeetham, Faridabad


### 📖 Course Requirements
This project was developed in partial fulfillment of the requirements for:
- **Deep Learning for Biomedical Data**
- **Deep Learning**

### 🎯 Academic Objectives
- Demonstrate practical application of transfer learning in medical imaging
- Address real-world healthcare challenges using AI/ML techniques
- Implement and compare state-of-the-art CNN architectures
- Analyze performance metrics relevant to medical diagnostics
- Contribute to research in automated medical image analysis

---

## 📋 Table of Contents
-
- [🌟 Overview](#-overview)
- [🎓 Academic Context](#-academic-context)
- [✨ Key Features](#-key-features)
- [📊 Dataset](#-dataset)
- [🏗️ Model Architecture](#️-model-architecture)
- [🚀 Results](#-results)
- [⚙️ Installation](#️-installation)
- [🔧 Usage](#-usage)
- [📈 Performance Metrics](#-performance-metrics)
- [🔬 Technical Details](#-technical-details)
- [🎯 Future Work](#-future-work)
- [👥 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

This project develops a deep learning pipeline for classifying breast ultrasound images into benign, malignant, and normal categories to facilitate early breast cancer detection. Utilizing transfer learning with pre-trained models (VGG16, ResNet50, and InceptionV3), the pipeline achieves a test accuracy of 73.48% using the InceptionV3 model, with a sensitivity of 61.29% for malignant cases, crucial for medical diagnostics.

The project addresses challenges such as a small dataset size (780 images), class imbalance, and high sensitivity requirements for medical applications. It leverages data augmentation, class weighting, and cloud-based GPU computing (Google Colab) to ensure robust performance. The pipeline is designed for potential telemedicine applications, enabling remote diagnosis in resource-limited settings.

This repository contains the source code, documentation, and instructions to reproduce the results, as submitted for partial fulfillment of the M.Tech. degree in Artificial Intelligence and Data Science at Amrita Vishwa Vidyapeetham, Faridabad.

### 🎯 Project Goals
- **Early Detection**: Enable timely identification of breast cancer through automated image analysis
- **Accessibility**: Provide diagnostic capabilities in underserved regions with limited medical expertise  
- **Accuracy**: Achieve reliable classification with high sensitivity for malignant cases
- **Scalability**: Create a deployable solution for telemedicine applications

---

## ✨ Key Features

🔬 **Multi-Class Classification**: Distinguishes between benign, malignant, and normal breast tissue  
🧠 **Transfer Learning**: Utilizes pre-trained CNN models (VGG16, ResNet50, InceptionV3)  
⚖️ **Class Imbalance Handling**: Implements weighted loss functions to address dataset imbalance  
🔄 **Data Augmentation**: Enhances model robustness through image transformations  
📊 **Comprehensive Evaluation**: Detailed performance metrics including sensitivity and specificity  
☁️ **Cloud-Ready**: Optimized for Google Colab with GPU acceleration  

---

## 📊 Dataset

### Dataset Overview
- **Source**: [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) (Kaggle)
- **Total Images**: 780 ultrasound images
- **Image Format**: PNG/JPG files
- **Resolution**: Variable (resized to model-specific dimensions)

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| 🟢 Benign | 437 | 56.0% |
| 🔴 Malignant | 210 | 27.0% |
| ⚪ Normal | 133 | 17.0% |

### Data Split Strategy
- **Training**: 70% (546 images) - with augmentation
- **Validation**: 15% (117 images) - for hyperparameter tuning
- **Testing**: 15% (117 images) - for final evaluation

---

## 🏗️ Model Architecture

### Pre-trained Models
We implemented and compared three state-of-the-art CNN architectures:

#### 1. 🧠 VGG16
- **Layers**: 16 layers
- **Parameters**: 138M
- **Input Size**: 224×224×3
- **Strengths**: Simple architecture, proven performance

#### 2. 🔗 ResNet50  
- **Layers**: 50 layers with residual connections
- **Parameters**: 25M
- **Input Size**: 224×224×3
- **Strengths**: Addresses vanishing gradient problem

#### 3. 🎯 InceptionV3 (Best Performer)
- **Layers**: 48 layers with inception modules
- **Parameters**: 24M  
- **Input Size**: 299×299×3
- **Strengths**: Multi-scale feature extraction

### Custom Classification Head
```
GlobalAveragePooling2D → Dense(512, ReLU) → BatchNormalization → Dropout(0.5) → Dense(3, Softmax)
```

---

## 🚀 Results

### 🏆 Best Model Performance (InceptionV3 with RMSprop)

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **73.48%** |
| **Test Loss** | 0.7859 |
| **Training Time** | 3,015 seconds |

### 📊 Class-wise Performance

| Class | Sensitivity | Specificity | F1-Score |
|-------|------------|-------------|----------|
| **Benign** | 86.36% | 74.51% | 0.8382 |
| **Malignant** | **61.29%** | **91.86%** | **0.6667** |
| **Normal** | 60.00% | 90.72% | 0.5854 |

### 🔄 Model Comparison

| Model | Accuracy | Loss | Training Time |
|-------|----------|------|---------------|
| VGG16 | 64.96% | 0.9251 | 5,349s |
| ResNet50 | 74.36% | 0.7554 | 1,955s |
| **InceptionV3** | **75.21%** | **0.7460** | **3,015s** |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Google Colab Pro (for cloud execution)

### Required Dependencies
```bash
pip install tensorflow==2.15
pip install keras==2.15
pip install scikit-learn==1.4
pip install pandas==2.2
pip install matplotlib==3.8
pip install seaborn==0.13
pip install Pillow
```

### Clone Repository
```bash
git clone (https://github.com/soumnemishra/Breast_cancer_detection.git)
cd breast-cancer-classification
```

---

## 🔧 Usage

### 1. Dataset Preparation
```python
# Download dataset from Kaggle
# Organize in directory structure:
# dataset/
# ├── benign/
# ├── malignant/
# └── normal/
```

### 2. Training Models
```python
# Run the complete pipeline
python train_models.py

# Or use individual components
from src.preprocessing import load_and_preprocess_data
from src.models import build_model, train_model
from src.evaluation import evaluate_model

# Load data
train_gen, val_gen, test_gen = load_and_preprocess_data()

# Build and train model
model = build_model('inception_v3')
history = train_model(model, train_gen, val_gen)

# Evaluate performance
results = evaluate_model(model, test_gen)
```

### 3. Making Predictions
```python
# Load trained model
model = tf.keras.models.load_model('models/inception_v3_model.keras')

# Preprocess new image
image = preprocess_image('path/to/ultrasound_image.jpg')

# Make prediction
prediction = model.predict(image)
class_names = ['benign', 'malignant', 'normal']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")
```

---

## 📈 Performance Metrics

### Confusion Matrix (InceptionV3)
```
           Predicted
Actual    Ben  Mal  Nor
Benign    57    8    1   (86.36% sensitivity)
Malignant  7   19    5   (61.29% sensitivity)  
Normal     6    2   12   (60.00% sensitivity)
```

### Key Medical Metrics
- **Malignant Sensitivity**: 61.29% (crucial for cancer detection)
- **Malignant Specificity**: 91.86% (low false positive rate)
- **Overall Accuracy**: 73.48% (competitive for small dataset)

---

## 🔬 Technical Details

### Data Preprocessing
- **Normalization**: ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
- **Augmentation**: Rotation (±20°), horizontal flip, zoom (±10%), brightness adjustment
- **Resizing**: Model-specific dimensions (224×224 or 299×299)

### Training Configuration
- **Optimizer**: RMSprop (learning_rate=0.0001)
- **Loss Function**: Categorical Crossentropy with label smoothing (0.1)
- **Batch Size**: 8-16 (model-dependent)
- **Epochs**: 20 with early stopping (patience=5)
- **Class Weights**: Balanced (Benign: 0.59, Malignant: 1.24, Normal: 1.96)

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (12GB+ recommended)
- **Storage**: 2GB for models and datasets

---

## 🎯 Future Work

### 🚀 Planned Enhancements
- [ ] **Dataset Expansion**: Collaborate with medical institutions for larger, diverse datasets
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy
- [ ] **Explainable AI**: Implement Grad-CAM for visualization of model decision-making
- [ ] **Mobile Deployment**: Optimize models for mobile and edge devices
- [ ] **Clinical Integration**: Develop API for integration with hospital systems
- [ ] **Multi-modal Analysis**: Incorporate patient history and clinical data

### 🔬 Research Directions
- [ ] **Federated Learning**: Train models across multiple hospitals while preserving privacy
- [ ] **Semi-supervised Learning**: Leverage unlabeled ultrasound images
- [ ] **Domain Adaptation**: Improve generalization across different ultrasound machines
- [ ] **Real-time Processing**: Optimize for live ultrasound analysis

---

## 👥 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- 🐛 Bug fixes and code improvements
- 📊 New evaluation metrics and visualizations
- 🧠 Novel model architectures
- 📝 Documentation improvements
- 🧪 Additional preprocessing techniques

---



## 🙏 Acknowledgments

### 👨‍🎓 Author
**Soumen Mishra** 
M.Tech Student, Artificial Intelligence and Data Science  
School of Artificial Intelligence  
Amrita Vishwa Vidyapeetham, Faridabad  

### 🎯 Academic Supervisors
- **Dr. Lakshmi Mohandas** - Project Guide (Deep Learning)
- **Dr. Sakshi Ahuja** - Project Guide (Deep Learning for Biomedical Data)
- **Prof. Kamal Bijlani** - Dean, School of Artificial Intelligence

### 🏛️ Institution
**Amrita Vishwa Vidyapeetham, Faridabad**  
School of Artificial Intelligence  
*Fostering innovation and research in AI for healthcare*

### 🛠️ Technical Resources
- **Dataset**: [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) from Kaggle
- **Computing Platform**: Google Colab Pro for GPU acceleration
- **Development Frameworks**: TensorFlow 2.15, Keras 2.15, Scikit-learn 1.4
- **Visualization Tools**: Matplotlib 3.8, Seaborn 0.13

### 🌐 Open Source Community
Special gratitude to the open-source community for providing invaluable tools and resources that made this academic research possible.

### 📚 Academic Impact
This project represents a contribution to the growing body of research in AI-powered medical diagnostics, specifically focusing on breast cancer detection using deep learning techniques. The work demonstrates the potential for automated systems to assist healthcare providers, particularly in resource-limited settings.

---


  
  <p><em>Academic research by Soumen Mishra - M.Tech AI & DS, Amrita Vishwa Vidyapeetham</em></p>
  <p><em>Made with ❤️ for advancing medical AI and improving healthcare outcomes</em></p>
</div>
