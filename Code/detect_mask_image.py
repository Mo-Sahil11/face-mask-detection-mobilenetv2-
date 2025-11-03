#!/usr/bin/env python3
"""
Face Mask Detection - Image Processing
=======================================
Detects face masks in static images using OpenCV DNN for face detection
and a trained MobileNetV2 model for mask classification.

Usage:
    python detect_mask_image.py --image path/to/image.jpg --model mask_detector.model

Output:
    - Green bounding box: Mask detected
    - Red bounding box: No mask detected
    - Confidence score displayed for each detection
"""

# Standard library imports
import argparse
import os

# Third-party imports
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ============================================================================
# Configuration
# ============================================================================

# Parse command-line arguments
ap = argparse.ArgumentParser(
    description="Detect face masks in static images",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("-i", "--image", required=True,
                help="path to input image file")
ap.add_argument("-f", "--face", type=str, default="face_detector",
                help="path to face detector model directory (default: face_detector)")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to trained face mask detector model (default: mask_detector.model)")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak face detections (default: 0.5)")
args = vars(ap.parse_args())

# ============================================================================
# Load Face Detector Model
# ============================================================================

print("[INFO] Loading face detector model...")
print(f"[INFO] Face detector directory: {args['face']}")

# Construct paths to face detector model files
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])

# Validate face detector files exist
if not os.path.exists(prototxtPath):
    raise ValueError(
        f"[ERROR] Face detector prototxt file not found: {prototxtPath}\n"
        f"Please run 'python download_face_detector.py' to download the required files."
    )

if not os.path.exists(weightsPath):
    raise ValueError(
        f"[ERROR] Face detector weights file not found: {weightsPath}\n"
        f"Please run 'python download_face_detector.py' to download the required files."
    )

# Load OpenCV DNN face detector
# Uses Single Shot Detector (SSD) with ResNet-10 backbone
net = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[INFO] Face detector model loaded successfully")

# ============================================================================
# Load Mask Detection Model
# ============================================================================

print("[INFO] Loading face mask detection model...")
print(f"[INFO] Model path: {args['model']}")

# Validate model file exists
if not os.path.exists(args["model"]):
    raise ValueError(
        f"[ERROR] Face mask detector model not found: {args['model']}\n"
        f"Please train the model first using 'train_mask_detector.py'."
    )

# Load trained MobileNetV2 mask classifier
model = load_model(args["model"])
print("[INFO] Mask detection model loaded successfully")

# ============================================================================
# Load and Process Input Image
# ============================================================================

print(f"[INFO] Loading input image: {args['image']}")

# Validate image file exists
if not os.path.exists(args["image"]):
    raise ValueError(f"[ERROR] Input image not found: {args['image']}")

# Read image from disk
image = cv2.imread(args["image"])
if image is None:
    raise ValueError(f"[ERROR] Could not load image: {args['image']}\n"
                    f"Please check if the file is a valid image format.")

# Store original for reference and get dimensions
orig = image.copy()
(h, w) = image.shape[:2]
print(f"[INFO] Image dimensions: {w}x{h}")

# ============================================================================
# Face Detection
# ============================================================================

print("[INFO] Detecting faces in image...")

# Create blob from image for DNN input
# blobFromImage parameters:
#   - image: input image
#   - 1.0: scaling factor
#   - (300, 300): spatial size for network input
#   - (104.0, 177.0, 123.0): mean subtraction values (BGR order) for normalization
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass blob through face detection network
net.setInput(blob)
detections = net.forward()

print(f"[INFO] Found {detections.shape[2]} potential face detections")

# ============================================================================
# Mask Classification & Visualization
# ============================================================================

face_count = 0
mask_count = 0
no_mask_count = 0

# Process each detected face
for i in range(0, detections.shape[2]):
    # Extract confidence score for this detection
    confidence = detections[0, 0, i, 2]
    
    # Filter weak detections based on confidence threshold
    if confidence > args["confidence"]:
        face_count += 1
        
        # Calculate bounding box coordinates
        # detections format: [batch, class_id, detection_num, [bbox_coords, confidence]]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Ensure bounding box is within image boundaries
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w - 1, endX)
        endY = min(h - 1, endY)
        
        # Extract face region of interest (ROI)
        face = image[startY:endY, startX:endX]
        
        # Skip if face ROI is too small (likely false detection)
        if face.shape[0] < 20 or face.shape[1] < 20:
            continue
        
        # Preprocess face for mask classification
        # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (224x224 for MobileNetV2)
        face_resized = cv2.resize(face_rgb, (224, 224))
        
        # Convert to array and preprocess
        face_array = img_to_array(face_resized)
        face_preprocessed = preprocess_input(face_array)
        face_batch = np.expand_dims(face_preprocessed, axis=0)
        
        # Classify mask/no-mask
        # Model outputs: [mask_probability, no_mask_probability]
        predictions = model.predict(face_batch, verbose=0)
        (mask_prob, without_mask_prob) = predictions[0]
        
        # Determine classification result
        label = "Mask" if mask_prob > without_mask_prob else "No Mask"
        confidence_score = max(mask_prob, without_mask_prob)
        
        # Update counters
        if label == "Mask":
            mask_count += 1
        else:
            no_mask_count += 1
        
        # Set visualization color (green for mask, red for no mask)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Format label with confidence percentage
        label_text = "{}: {:.1f}%".format(label, confidence_score * 100)
        
        # Draw bounding box around face
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        # Draw label above bounding box
        # Adjust label position to avoid going off-screen
        label_y = startY - 10 if startY - 10 > 10 else startY + 30
        cv2.putText(image, label_text, (startX, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

# ============================================================================
# Display Results
# ============================================================================

print("\n" + "="*60)
print("DETECTION RESULTS")
print("="*60)
print(f"Total faces detected: {face_count}")
print(f"  - With mask: {mask_count}")
print(f"  - Without mask: {no_mask_count}")
print(f"Confidence threshold: {args['confidence']}")
print("="*60)

# Display the annotated image
print("[INFO] Displaying results (press any key to close)...")
cv2.imshow("Face Mask Detection Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("[INFO] Detection complete!")
