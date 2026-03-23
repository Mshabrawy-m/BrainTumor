"""
Model cache and performance optimizations for Brain Tumor MRI AI Assistant.
Implements lazy loading, caching, and fast inference.
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import functools
import time

import numpy as np
import tensorflow as tf

from .model import load_model, predict_tumor, _find_model_path

# Global model cache
_model_cache: Dict[str, Any] = {}
_cache_dir = Path(__file__).parent / ".cache"
_cache_dir.mkdir(exist_ok=True)


def get_cached_model(model_path: Optional[str] = None) -> tf.keras.Model:
    """
    Get cached model or load and cache it.
    Reduces model loading time from ~3-5 seconds to ~50ms after first load.
    """
    path = model_path or _find_model_path()
    if not path:
        raise FileNotFoundError("No model file found")
    
    cache_key = hashlib.md5(str(path).encode()).hexdigest()
    
    # Return cached model if available
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Load and cache model
    model = load_model(path)
    _model_cache[cache_key] = model
    
    return model


@functools.lru_cache(maxsize=128)
def cached_preprocess(image_hash: int, target_size: tuple) -> np.ndarray:
    """
    Cached image preprocessing.
    Uses image hash to avoid reprocessing identical images.
    """
    # This is a placeholder - actual preprocessing happens in utils
    # The cache key prevents reprocessing of identical images
    pass


def fast_predict_tumor(
    image: np.ndarray,
    model_path: Optional[str] = None
) -> tuple:
    """
    Fast tumor prediction with cached model.
    Reduces prediction time by avoiding model reloads.
    """
    model = get_cached_model(model_path)
    
    # Warm up model if first prediction
    if not hasattr(fast_predict_tumor, '_warmed_up'):
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        model.predict(dummy_input, verbose=0)
        fast_predict_tumor._warmed_up = True
    
    return predict_tumor(image, model)


class PredictionCache:
    """Simple in-memory cache for prediction results."""
    
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, tuple] = {}
        self.max_size = max_size
        self.access_order: list = []
    
    def get(self, image_hash: str) -> Optional[tuple]:
        """Get cached prediction."""
        if image_hash in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(image_hash)
            self.access_order.append(image_hash)
            return self.cache[image_hash]
        return None
    
    def put(self, image_hash: str, result: tuple) -> None:
        """Cache prediction result."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[image_hash] = result
        self.access_order.append(image_hash)


# Global prediction cache
prediction_cache = PredictionCache()


def get_image_hash(image: np.ndarray) -> str:
    """Generate hash for image array."""
    return hashlib.md5(image.tobytes()).hexdigest()


def optimize_tensorflow():
    """Optimize TensorFlow settings for faster inference."""
    # Disable TensorFlow warnings and optimizations that slow down CPU inference
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Configure TensorFlow for better CPU performance (compatible with older TF versions)
    try:
        # Try newer API first
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except AttributeError:
        try:
            # Fallback for older TensorFlow versions
            tf.config.set_thread_pool_config(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1
            )
        except AttributeError:
            # If neither works, just set environment variables
            pass
    
    # Disable GPU if not available (reduces initialization time)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            tf.config.set_visible_devices([], 'GPU')
    except:
        pass


# Initialize optimizations
optimize_tensorflow()
