# 🎭 Face Mask Detection System

A deep learning-based face mask detection system built with TensorFlow/Keras and OpenCV. This project implements real-time mask detection using transfer learning with MobileNetV2 architecture, designed for both image analysis and live video streams.

## 🌟 Features

- **Real-time Detection**: Live video stream processing with OpenCV
- **Image Analysis**: Batch processing for static images
- **High Accuracy**: 92.53% accuracy using fine-tuned MobileNetV2
- **Optimized Performance**: Lightweight model suitable for edge devices
- **Easy Setup**: Automated face detector model download script
- **Comprehensive Error Handling**: User-friendly error messages and validation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd Face-mask-Detection-using-Transfer-learning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download face detector models:**
   ```bash
   cd Code
   python download_face_detector.py
   ```

4. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; import cv2; print('✓ Setup complete!')"
   ```

## 📖 Usage

### Training Your Model

Prepare your dataset with the following structure:
```
dataset/
├── with_mask/
│   ├── image1.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    └── ...
```

Train the model:
```bash
cd Code
python train_mask_detector.py --dataset ../dataset --model mask_detector.model --plot training_history.png
```

**Training Parameters** (configurable in code):
- Learning Rate: 1e-4
- Epochs: 20
- Batch Size: 32
- Train/Test Split: 80/20

### Detecting Masks in Images

```bash
cd Code
python detect_mask_image.py --image ../examples/example_01.png --model mask_detector.model
```

**Output:**
- 🟢 Green box = Mask detected
- 🔴 Red box = No mask detected

### Real-time Video Detection

```bash
cd Code
python detect_mask_video.py --model mask_detector.model --confidence 0.5
```

**Controls:**
- Press `q` to quit
- Adjust `--confidence` for detection sensitivity (default: 0.5)

## 🏗️ Architecture & Methodology

### Technical Stack
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Face Detection**: OpenCV DNN with ResNet-10 SSD backbone
- **Classification**: Binary classifier (with_mask / without_mask)
- **Optimizer**: Adam with polynomial learning rate decay
- **Loss Function**: Binary Cross Entropy

### Model Architecture

The model uses transfer learning by:
1. Loading pre-trained MobileNetV2 (excluding top layers)
2. Adding custom classification head:
   - Average Pooling (7x7)
   - Dense layer (128 units, ReLU)
   - Dropout (0.5)
   - Output layer (2 units, Softmax)

### Data Pipeline

```
Raw Images → Preprocessing → Data Augmentation → Training → Model Evaluation
```

**Augmentation Techniques:**
- Rotation (±20°)
- Zoom (0.85-1.0x)
- Shifts (width/height)
- Shear transformation
- Horizontal flip

## 📊 Performance Metrics

- **Accuracy**: 92.53%
- **F1 Score**: 0.93
- **True Positives**: 941
- **True Negatives**: 1103
- **False Positives**: 2
- **False Negatives**: 163

## 🔧 Project Structure

```
Face-mask-Detection-using-Transfer-learning/
├── Code/
│   ├── train_mask_detector.py      # Model training script
│   ├── detect_mask_image.py         # Image detection
│   ├── detect_mask_video.py         # Real-time video detection
│   ├── download_face_detector.py    # Face detector model downloader
│   └── face_detector/               # Face detection models (auto-downloaded)
├── Dataset-example/                 # Example dataset structure
├── examples/                         # Sample test images
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🎯 Use Cases

- **Public Safety**: Monitoring mask compliance in public spaces
- **Access Control**: Entry point verification systems
- **Analytics**: Mask usage statistics and reporting
- **Research**: Public health studies and data collection

## 🛠️ Customization

### Adjusting Detection Confidence
Modify the confidence threshold in detection scripts:
```python
# Lower = more sensitive, Higher = more strict
confidence = 0.5  # Default value
```

### Training Parameters
Edit hyperparameters in `train_mask_detector.py`:
```python
INIT_LR = 1e-4    # Learning rate
EPOCHS = 20       # Training epochs
BS = 32           # Batch size
```

### Model Architecture
Customize the classification head in `train_mask_detector.py`:
```python
# Modify dense layer units, dropout rate, etc.
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
```

## 📝 Notes

- The model requires a GPU for optimal training performance (training takes ~6-7 hours on CPU)
- For best results, ensure balanced dataset distribution
- Real-time performance depends on system hardware (optimized for modern CPUs/GPUs)

## 🔗 Dataset Sources

This project uses datasets from:
1. [Kaggle Medical Mask Dataset](https://www.kaggle.com/mloey1/medical-face-mask-detection-dataset)
2. [Real-World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)
3. [Prajna Bhandary Dataset](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)

## 📄 License

MIT License - Feel free to use this project for your own applications!

---

**Built with TensorFlow, Keras, and OpenCV**
