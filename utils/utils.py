"""
Image preprocessing utilities for Brain Tumor MRI AI Assistant.
Uses OpenCV for resize, normalize, and format conversion.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# Target input size for the TensorFlow model
TARGET_SIZE: Tuple[int, int] = (224, 224)


def preprocess_mri_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE
) -> Optional[np.ndarray]:
    """
    Preprocess an MRI image for model inference.
    
    Steps:
    1. Resize to target size (224x224)
    2. Convert to RGB if grayscale
    3. Normalize pixel values to 0-1
    4. Expand dimensions for model input (batch dimension)
    
    Args:
        image: Input image as numpy array (BGR from OpenCV or RGB)
        target_size: Output dimensions (width, height)
        
    Returns:
        Preprocessed image with shape (1, height, width, 3) or None on error
    """
    try:
        # Convert to numpy array if needed (e.g., from PIL)
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle grayscale - convert to BGR then RGB for consistency
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Convert BGR to RGB if needed (OpenCV reads as BGR)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Expand dimensions for batch: (H, W, C) -> (1, H, W, C)
        image = np.expand_dims(image, axis=0)
        
        return image
        
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}") from e


def load_image_from_file(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path using OpenCV.
    
    Args:
        file_path: Path to image file (JPG, PNG, etc.)
        
    Returns:
        Image as numpy array or None on error
    """
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image from {file_path}")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}") from e


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load an image from bytes (e.g., uploaded file).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Image as numpy array or None on error
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}") from e
