# 🧬 Breast Cancer Detection using Deep Learning on Ultrasound Images

Breast cancer is one of the most common causes of death among women worldwide. **Early detection** significantly improves survival rates. In this project, we leverage deep learning techniques to classify **breast ultrasound images** into three categories: **normal**, **benign**, and **malignant**.

## 📊 Dataset

This project uses the **Breast Ultrasound Images Dataset**, publicly available and curated by Al-Dhabyani et al.

- **Total Patients**: 600 women (ages 25–75)
- **Total Images**: 780 PNG images
- **Image Size**: ~500x500 pixels
- **Classes**: 
  - 🟢 Normal
  - 🟡 Benign
  - 🔴 Malignant
- **Includes**: Original images and corresponding ground truth masks
- **Year of Collection**: 2018

> 📚 **If you use this dataset, please cite:**
>
> *Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863.*
> [DOI: 10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)

## 🧠 Methods Used

- Convolutional Neural Networks (CNNs)
- TensorFlow / Keras framework
- Data augmentation (e.g., flipping, rotation)
- Categorical Crossentropy loss
- Softmax activation for multi-class classification
- Confusion matrix, classification report for evaluation

## 📂 Project Files

- `breast_cancer_detection.ipynb`: Main notebook containing the complete workflow (loading data, preprocessing, model building, training, evaluation).
- `README.md`: Project overview and setup guide.

## 🚀 How to Run

1. Clone the repository:
   ```bash
 [  git clone https://github.com/your-username/breast_cancer_detection.git](https://github.com/soumnemishra/Breast_cancer_detection.git)

