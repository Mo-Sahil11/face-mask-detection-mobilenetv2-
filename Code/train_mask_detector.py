#!/usr/bin/env python3
"""
Face Mask Detector - Training Script
=====================================
This script trains a face mask detection model using transfer learning
with MobileNetV2 architecture.

Usage:
    python train_mask_detector.py --dataset dataset --model mask_detector.model --plot plot.png
"""

# Standard library imports
import argparse
import os

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# TensorFlow/Keras imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.utils import to_categorical

# ============================================================================
# Configuration & Hyperparameters
# ============================================================================

# Parse command-line arguments
ap = argparse.ArgumentParser(
    description="Train a face mask detection model using MobileNetV2 transfer learning"
)
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (should contain 'with_mask' and 'without_mask' folders)")
ap.add_argument("-p", "--plot", type=str, default="training_history.png",
                help="path to output training history plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to output trained model file")
args = vars(ap.parse_args())

# Training hyperparameters - tuned for optimal performance
INIT_LR = 1e-4      # Initial learning rate (low for fine-tuning)
EPOCHS = 20          # Number of training epochs
BS = 32              # Batch size (adjust based on GPU memory)

# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

print("[INFO] Loading and preprocessing images...")
print("[INFO] Dataset path:", args["dataset"])

# Validate dataset path
if not os.path.exists(args["dataset"]):
    raise ValueError(f"[ERROR] Dataset path does not exist: {args['dataset']}")

# Gather all image paths from the dataset directory
imagePaths = list(paths.list_images(args["dataset"]))
print(f"[INFO] Found {len(imagePaths)} images in dataset")

if len(imagePaths) == 0:
    raise ValueError(f"[ERROR] No images found in dataset directory: {args['dataset']}")

# Initialize lists to store processed images and labels
data = []
labels = []

# Process each image
print("[INFO] Processing images...")
for (i, imagePath) in enumerate(imagePaths):
    # Extract class label from directory structure (with_mask/without_mask)
    label = imagePath.split(os.path.sep)[-2]
    
    # Load image and resize to MobileNetV2 input size (224x224)
    # MobileNetV2 expects 224x224 RGB images
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    
    # Preprocess image for MobileNetV2 (normalizes pixel values)
    image = preprocess_input(image)
    
    # Store processed image and label
    data.append(image)
    labels.append(label)
    
    # Progress indicator
    if (i + 1) % 1000 == 0:
        print(f"[INFO] Processed {i + 1}/{len(imagePaths)} images...")

print("[INFO] Image preprocessing complete")

# Convert to NumPy arrays for efficient processing
data = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"[INFO] Dataset shape: {data.shape}")
print(f"[INFO] Labels distribution:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  - {u}: {c} images")

# ============================================================================
# Label Encoding
# ============================================================================

print("[INFO] Encoding labels...")
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

print(f"[INFO] Number of classes: {labels.shape[1]}")

# ============================================================================
# Train/Test Split
# ============================================================================

print("[INFO] Splitting dataset into training and testing sets...")
(trainX, testX, trainY, testY) = train_test_split(
    data, labels,
    test_size=0.20,      # 80% train, 20% test
    stratify=labels,      # Maintain class distribution
    random_state=42       # Reproducibility
)

print(f"[INFO] Training samples: {trainX.shape[0]}")
print(f"[INFO] Testing samples: {testX.shape[0]}")

# ============================================================================
# Data Augmentation
# ============================================================================

print("[INFO] Setting up data augmentation...")
# Data augmentation helps prevent overfitting and improves generalization
# by artificially increasing dataset size with transformed images
aug = ImageDataGenerator(
    rotation_range=20,        # Random rotation ±20 degrees
    zoom_range=0.15,          # Random zoom 85-115%
    width_shift_range=0.2,    # Random horizontal shift ±20%
    height_shift_range=0.2,   # Random vertical shift ±20%
    shear_range=0.15,         # Random shearing transformation
    horizontal_flip=True,     # Random horizontal flipping
    fill_mode="nearest"       # Fill pixels with nearest neighbor
)

print("[INFO] Data augmentation configured")

# ============================================================================
# Model Architecture
# ============================================================================

print("[INFO] Building model architecture...")

# Load MobileNetV2 base model (pre-trained on ImageNet)
# We exclude the top (classification) layer since we'll add our own
baseModel = MobileNetV2(
    weights="imagenet",        # Use pre-trained ImageNet weights
    include_top=False,         # Don't include classification layer
    input_tensor=Input(shape=(224, 224, 3))  # Input image shape
)

print("[INFO] MobileNetV2 base model loaded")

# Build custom classification head
# This replaces the original MobileNetV2 classification layer
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)  # Global average pooling
headModel = Flatten(name="flatten")(headModel)             # Flatten to 1D
headModel = Dense(128, activation="relu")(headModel)      # Dense layer with ReLU
headModel = Dropout(0.5)(headModel)                        # Dropout for regularization
headModel = Dense(2, activation="softmax")(headModel)     # Binary classification output

# Combine base model with custom head
model = Model(inputs=baseModel.input, outputs=headModel)

print("[INFO] Model architecture created")
print(f"[INFO] Total trainable parameters: {model.count_params():,}")

# Freeze base model layers (transfer learning approach)
# We only train the custom head initially, keeping MobileNetV2 weights frozen
print("[INFO] Freezing base model layers...")
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] Base model frozen - only classification head will be trained")

# ============================================================================
# Model Compilation
# ============================================================================

print("[INFO] Compiling model...")

# Calculate learning rate schedule
# Polynomial decay: gradually reduce learning rate during training
steps_per_epoch = len(trainX) // BS
total_steps = EPOCHS * steps_per_epoch

lr_schedule = PolynomialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=total_steps,
    end_learning_rate=INIT_LR / EPOCHS  # End at 1/20th of initial rate
)

# Use Adam optimizer with learning rate schedule
opt = Adam(learning_rate=lr_schedule)

# Compile model with binary cross-entropy loss (suitable for binary classification)
model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

print("[INFO] Model compiled successfully")

# ============================================================================
# Training
# ============================================================================

print("[INFO] Starting training...")
print(f"[INFO] Training parameters:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch size: {BS}")
print(f"  - Steps per epoch: {steps_per_epoch}")
print(f"  - Initial learning rate: {INIT_LR}")

# Train the model
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),  # Training data with augmentation
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),            # Validation set (no augmentation)
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    verbose=1
)

print("[INFO] Training completed!")

# ============================================================================
# Model Evaluation
# ============================================================================

print("[INFO] Evaluating model on test set...")
predIdxs = model.predict(testX, batch_size=BS)

# Get predicted class indices
predIdxs = np.argmax(predIdxs, axis=1)

# Print detailed classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(
    testY.argmax(axis=1),
    predIdxs,
    target_names=lb.classes_
))
print("="*60)

# ============================================================================
# Save Model & Training History
# ============================================================================

# Save trained model
print(f"[INFO] Saving model to {args['model']}...")
model.save(args["model"], save_format="h5")
print("[INFO] Model saved successfully!")

# Plot training history
print(f"[INFO] Generating training history plot...")
N = EPOCHS
plt.style.use("ggplot")
plt.figure(figsize=(12, 6))

# Plot training loss and accuracy
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss", linewidth=2)
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss", linewidth=2)
plt.title("Training Loss", fontsize=14, fontweight='bold')
plt.xlabel("Epoch #", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(loc="upper right")

plt.subplot(1, 2, 2)
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc", linewidth=2)
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc", linewidth=2)
plt.title("Training Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Epoch #", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(args["plot"], dpi=150, bbox_inches='tight')
print(f"[INFO] Training plot saved to {args['plot']}")

print("\n" + "="*60)
print("[SUCCESS] Training pipeline completed!")
print("="*60)
print(f"Model: {args['model']}")
print(f"Plot: {args['plot']}")
print("="*60)
