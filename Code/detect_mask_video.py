#!/usr/bin/env python3
"""
Face Mask Detection - Real-time Video Processing
=================================================
Detects face masks in live video stream from webcam using OpenCV DNN
for face detection and MobileNetV2 for mask classification.

Features:
    - Real-time processing with optimized batch inference
    - Live statistics display
    - Adjustable confidence thresholds
    - Press 'q' to quit

Usage:
    python detect_mask_video.py --model mask_detector.model --confidence 0.5
"""

# Standard library imports
import argparse
import os
import time

# Third-party imports
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ============================================================================
# Helper Function: Face Detection & Mask Prediction
# ============================================================================

def detect_and_predict_mask(frame, faceNet, maskNet, minConfidence):
    """
    Detect faces in a frame and classify whether they're wearing masks.
    
    This function performs batch processing for efficiency - detects all faces
    first, then makes a single batch prediction rather than one-by-one.
    
    Args:
        frame: Input frame from video stream (BGR format)
        faceNet: Loaded OpenCV DNN face detector
        maskNet: Loaded trained mask classification model
        minConfidence: Minimum confidence threshold for face detection
    
    Returns:
        tuple: (face_locations, predictions)
            - face_locations: List of (startX, startY, endX, endY) tuples
            - predictions: NumPy array of [mask_prob, no_mask_prob] for each face
    """
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Create blob from frame for DNN input
    # Using standard MobileNet preprocessing values
    blob = cv2.dnn.blobFromImage(
        frame, 
        1.0,              # Scale factor
        (300, 300),       # Network input size
        (104.0, 177.0, 123.0)  # Mean subtraction (BGR)
    )
    
    # Pass blob through face detection network
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # Initialize lists for face ROIs and their locations
    faces = []
    locs = []
    
    # Process each detection
    for i in range(0, detections.shape[2]):
        # Extract confidence score
        confidence = detections[0, 0, i, 2]
        
        # Filter detections below confidence threshold
        if confidence > minConfidence:
            # Calculate bounding box coordinates (normalized to pixel values)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure bounding box stays within frame boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)
            
            # Extract face region of interest
            face = frame[startY:endY, startX:endX]
            
            # Skip invalid face regions (too small)
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            
            # Preprocess face for mask classification
            # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (MobileNetV2: 224x224)
            face_resized = cv2.resize(face_rgb, (224, 224))
            
            # Convert to array and preprocess
            face_array = img_to_array(face_resized)
            face_preprocessed = preprocess_input(face_array)
            face_batch = np.expand_dims(face_preprocessed, axis=0)
            
            # Store preprocessed face and its location
            faces.append(face_batch)
            locs.append((startX, startY, endX, endY))
    
    # Batch prediction for efficiency (process all faces at once)
    # This is faster than predicting one-by-one in the loop above
    preds = []
    if len(faces) > 0:
        # Concatenate all face batches into single batch
        faces_batch = np.vstack(faces)
        
        # Single batch prediction (much faster than loop)
        preds = maskNet.predict(faces_batch, verbose=0)
    
    # Return face locations and predictions
    return (locs, preds)

# ============================================================================
# Configuration
# ============================================================================

# Parse command-line arguments
ap = argparse.ArgumentParser(
    description="Real-time face mask detection from webcam",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("-f", "--face", type=str, default="face_detector",
                help="path to face detector model directory (default: face_detector)")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to trained face mask detector model (default: mask_detector.model)")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak face detections (default: 0.5)")
args = vars(ap.parse_args())

# ============================================================================
# Load Models
# ============================================================================

print("[INFO] Loading models...")

# Load face detector
print(f"[INFO] Loading face detector from: {args['face']}")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])

# Validate face detector files
if not os.path.exists(prototxtPath):
    raise ValueError(
        f"[ERROR] Face detector prototxt not found: {prototxtPath}\n"
        f"Run 'python download_face_detector.py' to download required files."
    )
if not os.path.exists(weightsPath):
    raise ValueError(
        f"[ERROR] Face detector weights not found: {weightsPath}\n"
        f"Run 'python download_face_detector.py' to download required files."
    )

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[INFO] Face detector loaded")

# Load mask classifier
print(f"[INFO] Loading mask classifier from: {args['model']}")
if not os.path.exists(args["model"]):
    raise ValueError(
        f"[ERROR] Mask detector model not found: {args['model']}\n"
        f"Train model first using 'train_mask_detector.py'."
    )

maskNet = load_model(args["model"])
print("[INFO] Mask classifier loaded")

# ============================================================================
# Initialize Video Stream
# ============================================================================

print("[INFO] Starting video stream...")
print("[INFO] Press 'q' to quit")

# Initialize video stream (src=0 for default webcam)
vs = VideoStream(src=0).start()

# Allow camera sensor to warm up
time.sleep(2.0)

# Statistics tracking
frame_count = 0
total_faces = 0
total_masks = 0
total_no_masks = 0
fps_start_time = time.time()
fps_frame_count = 0

# ============================================================================
# Main Processing Loop
# ============================================================================

print("[INFO] Starting detection loop...")

try:
    while True:
        # Read frame from video stream
        frame = vs.read()
        
        # Resize frame for faster processing (max width 400px)
        # This balances speed and detection accuracy
        frame = imutils.resize(frame, width=400)
        
        # Detect faces and predict masks
        (locs, preds) = detect_and_predict_mask(
            frame, faceNet, maskNet, args["confidence"]
        )
        
        # Process each detected face
        for (box, pred) in zip(locs, preds):
            # Unpack bounding box coordinates
            (startX, startY, endX, endY) = box
            
            # Unpack predictions [mask_prob, no_mask_prob]
            (mask_prob, without_mask_prob) = pred
            
            # Determine classification
            label = "Mask" if mask_prob > without_mask_prob else "No Mask"
            confidence_score = max(mask_prob, without_mask_prob)
            
            # Update statistics
            total_faces += 1
            if label == "Mask":
                total_masks += 1
            else:
                total_no_masks += 1
            
            # Set visualization color
            # Green for mask, Red for no mask
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            # Format label with confidence
            label_text = "{}: {:.1f}%".format(label, confidence_score * 100)
            
            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Draw label
            cv2.putText(
                frame, label_text, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2
            )
        
        # Calculate and display FPS
        frame_count += 1
        fps_frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start_time
            fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Display statistics on frame
        stats_text = [
            f"Faces detected: {len(locs)}",
            f"Total processed: {total_faces}",
            f"With mask: {total_masks}",
            f"Without mask: {total_no_masks}",
        ]
        
        y_offset = 20
        for text in stats_text:
            cv2.putText(
                frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += 20
        
        # Display frame
        cv2.imshow("Face Mask Detection - Press 'q' to quit", frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    # Cleanup
    print("\n[INFO] Cleaning up...")
    print(f"[INFO] Session statistics:")
    print(f"  - Total frames processed: {frame_count}")
    print(f"  - Total faces detected: {total_faces}")
    print(f"  - With mask: {total_masks}")
    print(f"  - Without mask: {total_no_masks}")
    
    cv2.destroyAllWindows()
    vs.stop()
    print("[INFO] Cleanup complete")
