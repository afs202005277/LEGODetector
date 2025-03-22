# Computer Vision Project: LEGO Piece Detection and Counting

## Overview

This project focuses on detecting and counting LEGO pieces in images using convolutional neural networks (CNNs) and object detection models. The work is divided into two main tasks:

- Development and evaluation of a CNN-based approach for counting the number of LEGO pieces in an image.
- Object detection and segmentation of LEGO pieces using state-of-the-art deep learning models.

## Dataset

The dataset used for this project consists of **2,933 real images** sourced from a Kaggle publication, along with **2,600 additional images** obtained through data augmentation. The augmentation process involved:

1. **Extracting individual LEGO pieces** from single-piece images using XML annotations, preprocessing, and foreground segmentation.
2. **Generating synthetic images** by placing extracted pieces onto varied backgrounds to create more diverse training data.

The final dataset was split into:
- **80% training**
- **10% validation**
- **10% testing** (only real images were included in the test set)

## Task 2: Counting LEGO Pieces

### Approach

To count LEGO pieces in an image, two different approaches were explored:

1. **Custom Convolutional Neural Network (CNN)**  
   - Five convolutional layers followed by max-pooling layers.
   - A fully connected layer producing a single output (piece count).
   - Achieved moderate performance, balancing complexity and accuracy.

2. **Transfer Learning**  
   - Fine-tuned pre-trained models (VGG16, ResNet50, DenseNet, EfficientNet).
   - The final fully connected layer was modified for regression.
   - **EfficientNet** achieved the best performance.

### Training Procedure

- **Image Preprocessing**:
  - Resizing (520×390 for custom CNN, different sizes for pre-trained models).
  - Gaussian noise and flipping for data augmentation.
  - Normalization using ImageNet statistics.

- **Loss Function**:
  - Mean Squared Error (MSE) initially, switching to Mean Absolute Error (MAE) once the loss reached a threshold of 1.

- **Optimization**:
  - **Adam optimizer** (learning rate: 0.001 for custom CNN, 0.0001 for pre-trained models).
  - Training conducted for **100 epochs** for the custom CNN, **30 epochs** for pre-trained models.

---

## Task 3: Object Detection and Segmentation

### Object Detection

Two object detection models were tested:
1. **Faster R-CNN** (built from scratch using course material and online examples).
2. **YOLO (You Only Look Once)** (using the Ultralytics module for implementation).

#### Detection Results (Mean Average Precision - mAP)

| Model        | mAP  | Epochs |
|-------------|------|--------|
| Faster R-CNN | 0.88 | 20     |
| YOLO Nano    | 0.949 | 25     |
| **YOLO Small** | **0.955** | **25** |
| YOLO Medium  | 0.953 | 25     |
| YOLO Large   | 0.947 | 25     |

**Conclusion**: YOLO models significantly outperformed Faster R-CNN. The **best model was YOLO Small**, likely due to its balance between complexity and training efficiency.

### Segmentation

For the segmentation task, two approaches were explored:

1. **YOLO (multiple model sizes)**
2. **Custom segmentation model** (predicting a segmentation mask from the input image)

#### Segmentation Results (Mean Average Precision - mAP)

| Model        | mAP  | Epochs |
|-------------|------|--------|
| YOLO Nano   | 0.902 | 25     |
| **YOLO Small** | **0.913** | **25** |
| YOLO Medium | 0.898 | 25     |
| YOLO Large  | 0.862 | 25     |
| Custom Model | -    | 30     |

**Conclusion**: The **YOLO Small model outperformed all other models**. The custom segmentation model did not perform as well as expected, likely due to the need for a better loss function.

### Alternative Approaches

- **Mask R-CNN**: Could improve segmentation by integrating detection and segmentation in one framework.
- **Hybrid Approach**: Using object detection to obtain bounding boxes, then applying GrabCut for segmentation.

---

## Key Takeaways

- **EfficientNet** was the best model for counting LEGO pieces.
- **YOLO Small** performed best in both **detection and segmentation**.
- The dataset's augmentation and preprocessing significantly improved model robustness.
- Future improvements could include **Mask R-CNN** for segmentation and **longer training for larger models**.

