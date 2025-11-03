"""
Utility Functions for Face Mask Detection
==========================================
Helper functions for common operations across the project.
"""

import os
import time
import cv2
import numpy as np
from typing import Tuple, List


def validate_model_files(model_path: str, face_detector_dir: str = "face_detector") -> Tuple[bool, List[str]]:
    """
    Validate that all required model files exist.
    
    Args:
        model_path: Path to mask detection model
        face_detector_dir: Directory containing face detector files
    
    Returns:
        Tuple of (is_valid, list_of_missing_files)
    """
    missing_files = []
    
    # Check mask detection model
    if not os.path.exists(model_path):
        missing_files.append(f"Mask detector model: {model_path}")
    
    # Check face detector files
    prototxt_path = os.path.join(face_detector_dir, "deploy.prototxt")
    weights_path = os.path.join(face_detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(prototxt_path):
        missing_files.append(f"Face detector prototxt: {prototxt_path}")
    
    if not os.path.exists(weights_path):
        missing_files.append(f"Face detector weights: {weights_path}")
    
    return len(missing_files) == 0, missing_files


def draw_detection_box(
    image: np.ndarray,
    startX: int, startY: int, endX: int, endY: int,
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box and label on image.
    
    Args:
        image: Input image
        startX, startY, endX, endY: Bounding box coordinates
        label: Detection label
        confidence: Confidence score
        color: Box color (BGR format)
        thickness: Box line thickness
    
    Returns:
        Annotated image
    """
    # Draw bounding box
    cv2.rectangle(image, (startX, startY), (endX, endY), color, thickness)
    
    # Format label text
    label_text = f"{label}: {confidence:.1f}%"
    
    # Calculate label position (avoid going off-screen)
    label_y = startY - 10 if startY - 10 > 10 else startY + 30
    
    # Draw label
    cv2.putText(
        image, label_text,
        (startX, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness
    )
    
    return image


def preprocess_face_for_classification(face_roi: np.ndarray, target_size: Tuple[int, int] = (224, 224)):
    """
    Preprocess face ROI for mask classification.
    
    Args:
        face_roi: Face region of interest (BGR format)
        target_size: Target size for resizing
    
    Returns:
        Preprocessed face batch ready for model input
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    face_resized = cv2.resize(face_rgb, target_size)
    
    # Convert to array and preprocess
    face_array = img_to_array(face_resized)
    face_preprocessed = preprocess_input(face_array)
    face_batch = np.expand_dims(face_preprocessed, axis=0)
    
    return face_batch


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate frames per second.
    
    Args:
        start_time: Start time timestamp
        frame_count: Number of frames processed
    
    Returns:
        FPS value
    """
    elapsed = time.time() - start_time
    return frame_count / elapsed if elapsed > 0 else 0.0


def print_detection_summary(total_faces: int, masks: int, no_masks: int):
    """
    Print formatted detection summary.
    
    Args:
        total_faces: Total faces detected
        masks: Faces with masks
        no_masks: Faces without masks
    """
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    print(f"Total faces detected: {total_faces}")
    print(f"  ✓ With mask: {masks} ({masks/total_faces*100:.1f}%)" if total_faces > 0 else "  ✓ With mask: 0")
    print(f"  ✗ Without mask: {no_masks} ({no_masks/total_faces*100:.1f}%)" if total_faces > 0 else "  ✗ Without mask: 0")
    print("="*60)


def validate_image_path(image_path: str) -> bool:
    """
    Validate that image file exists and can be loaded.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        return False
    
    # Try to read image
    image = cv2.imread(image_path)
    return image is not None

