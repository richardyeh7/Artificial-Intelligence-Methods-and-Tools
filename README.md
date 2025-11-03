# Artificial-Intelligence-Methods-and-Tools

# Chest X-ray Pneumonia Classification

##  Project Overview

With the rapid growth of medical imaging, chest X-rays have become a vital tool for diagnosing lung diseases. However, manual interpretation is time-consuming and error-prone.  
In this project, we aim to **automate pneumonia detection** from chest X-ray images using deep learning.  
We implement and compare several CNN architectures — **VGG16/19, ResNet, and AlexNet** — on the **Kaggle Chest X-ray (Pneumonia)** dataset.  
Through data augmentation and early stopping, our **ResNet model achieved 90.87% accuracy and 93.04% F1-score**, outperforming other models and demonstrating the potential of AI-assisted diagnosis.


- **Dataset:** 5,863 X-ray images (JPEG), split into `train/`, `val/`, `test/`, each with `NORMAL/` and `PNEUMONIA/` folders. Data originates from Guangzhou Women and Children’s Medical Center.
- **Goal:** reduce reliance on manual reading and improve diagnostic speed by providing an automated screening model.
- **Best model:** **ResNet** with data augmentation and early stopping.
- **Best metrics:** **Accuracy = 90.87%**, **F1 = 93.04%** on the test set.

## 1. Dataset

- Source: Kaggle “Chest X-ray Images (Pneumonia)”
- 2 classes: `NORMAL`, `PNEUMONIA`
- Train/val/test already provided
- Images resized to 224×224 or 256→224 during augmentation

## 2. Models Compared

1. **VGG16 / VGG19**  
   - 3×3 conv blocks, deeper network
   - With Adam (1e-3) training was unstable → switched to SGD
   - Augmentation + early stopping improved VGG16, VGG19 did not surpass it

2. **ResNet (e.g. ResNet50)** 
   - Residual blocks + skip connections → better gradient flow
   - With augmentation and early stopping, this model achieved the best generalization
   - **Test Accuracy: 90.87%**
   - **Test F1: 93.04%**

3. **AlexNet**
   - Trains fine but slightly lower performance
   - **Accuracy ~84%, F1 ~88%**

4. **LeNet (dropped)**
   - Input resolution too small (28×28) for this medical dataset → information loss → not used in final comparison

## 📘 View in Google Colab

- [ResNet Notebook](https://colab.research.google.com/github/richardyeh7/Artificial-Intelligence-Methods-and-Tools/blob/main/ResNet.ipynb)
- [VGG16 Notebook](https://colab.research.google.com/github/richardyeh7/Artificial-Intelligence-Methods-and-Tools/blob/main/VGG16.ipynb)
- [AlexNet Notebook](https://colab.research.google.com/github/richardyeh7/Artificial-Intelligence-Methods-and-Tools/blob/main/AlexNet.ipynb#scrollTo=kS9EX6T_YxWZ)


## 3. Training Tricks

- **Data Augmentation** (PyTorch `transforms`):
  - `Resize(256,256)`
  - `RandomHorizontalFlip()`
  - `RandomRotation(20)`
  - `ColorJitter(...)`
  - `RandomResizedCrop(224)`
  - `ToTensor()`
- **Early Stopping**: monitor validation loss, stop if no improvement after N epochs → prevents overfitting observed in VGG/AlexNet
- **Optimizers**:
  - Adam(lr=1e-3) for faster convergence
  - SGD for VGG variants when Adam plateaued

## 4. Results

| Model   | Test Accuracy | F1 Score |
|---------|----------------|----------|
| ResNet  | **90.87%**     | **93.04%** |
| VGG16   | 82.21%         | 87.20% |
| AlexNet | 83.97%         | 88.61% |

ResNet confusion matrix shows:
- TP (pneumonia correctly detected): high
- Few false negatives (9 normal predicted)
- Fewer false positives than VGG

