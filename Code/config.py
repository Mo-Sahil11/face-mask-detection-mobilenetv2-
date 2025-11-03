"""
Configuration File for Face Mask Detection System
=================================================
Centralized configuration for all detection and training scripts.
Modify these values to customize behavior without editing individual scripts.
"""

# ============================================================================
# Training Configuration
# ============================================================================

TRAINING_CONFIG = {
    "initial_learning_rate": 1e-4,      # Starting learning rate
    "epochs": 20,                        # Number of training epochs
    "batch_size": 32,                    # Batch size for training
    "train_test_split": 0.20,             # Test set proportion (20%)
    "random_state": 42,                   # Random seed for reproducibility
    "input_size": (224, 224),            # Input image size (MobileNetV2 standard)
}

# ============================================================================
# Data Augmentation Configuration
# ============================================================================

AUGMENTATION_CONFIG = {
    "rotation_range": 20,                 # Random rotation in degrees (±20°)
    "zoom_range": 0.15,                   # Random zoom (85-115%)
    "width_shift_range": 0.2,             # Horizontal shift (±20%)
    "height_shift_range": 0.2,            # Vertical shift (±20%)
    "shear_range": 0.15,                  # Shear transformation
    "horizontal_flip": True,              # Random horizontal flip
    "fill_mode": "nearest",               # Fill mode for transformations
}

# ============================================================================
# Detection Configuration
# ============================================================================

DETECTION_CONFIG = {
    "face_detection_confidence": 0.5,    # Minimum confidence for face detection
    "mask_detection_confidence": 0.5,     # Minimum confidence for mask detection
    "face_detector_size": (300, 300),    # Face detector input size
    "mask_classifier_size": (224, 224),  # Mask classifier input size
    "video_frame_width": 400,            # Video frame processing width
}

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_CONFIG = {
    "base_model": "MobileNetV2",         # Base model for transfer learning
    "base_model_weights": "imagenet",     # Pre-trained weights
    "classification_head": {
        "dense_units": 128,               # Dense layer units
        "dropout_rate": 0.5,              # Dropout rate
        "activation": "relu",             # Activation function
        "output_units": 2,                # Output classes (mask/no_mask)
        "output_activation": "softmax",   # Output activation
    },
}

# ============================================================================
# Visualization Configuration
# ============================================================================

VISUALIZATION_CONFIG = {
    "mask_color": (0, 255, 0),           # Green for mask detected (BGR)
    "no_mask_color": (0, 0, 255),        # Red for no mask (BGR)
    "box_thickness": 2,                  # Bounding box line thickness
    "font_scale": 0.65,                  # Text font scale
    "font_thickness": 2,                  # Text font thickness
    "font": "FONT_HERSHEY_SIMPLEX",      # OpenCV font type
}

# ============================================================================
# Path Configuration
# ============================================================================

PATH_CONFIG = {
    "face_detector_dir": "face_detector",
    "default_model": "mask_detector.model",
    "default_plot": "training_history.png",
}

# ============================================================================
# Performance Configuration
# ============================================================================

PERFORMANCE_CONFIG = {
    "use_gpu": True,                     # Use GPU if available
    "num_threads": None,                 # Thread count (None = auto)
    "inter_op_parallelism": None,        # TensorFlow inter-op parallelism
    "intra_op_parallelism": None,        # TensorFlow intra-op parallelism
}

