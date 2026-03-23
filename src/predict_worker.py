"""
Standalone prediction worker - runs TensorFlow in separate process.
Prevents TensorFlow crashes from killing the Streamlit app.
Usage: python predict_worker.py <image_path>
Output: JSON to stdout {"predicted_class": "...", "confidence_score": 0.99}
"""

import json
import sys
from pathlib import Path

# Must be before TensorFlow
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Force CPU

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: predict_worker.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    try:
        # Add project root to path for imports
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.performance import fast_predict_tumor
        from utils.utils import load_image_from_file, preprocess_mri_image

        image = load_image_from_file(image_path)
        preprocessed = preprocess_mri_image(image)
        predicted_class, confidence_score = fast_predict_tumor(preprocessed)

        result = {
            "predicted_class": predicted_class,
            "confidence_score": float(confidence_score)
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
