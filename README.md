# Brain Tumor MRI AI Assistant

An AI-powered application for brain tumor MRI analysis. Upload an MRI scan, get tumor classification (glioma, meningioma, pituitary, no tumor), and ask questions via a specialized LLM assistant.

## Features

- **MRI Image Upload** – JPG/PNG support
- **Tumor Classification** – TensorFlow CNN predicts: glioma, meningioma, pituitary, no_tumor
- **Confidence Bar** – Visual confidence score display
- **Grad-CAM** – Optional visualization highlighting tumor regions
- **Chat History** – Memory of conversation
- **LLM Chat** – Groq (llama3-70b-8192) answers Brain Tumor MRI questions only
- **Model Caching** – TensorFlow model loaded once for server stability
- **Port Fallback** – Use `--server.port 8502` if 8501 is busy

## Quick Start

```bash
cd brain_tumor_mri_ai
pip install -r requirements.txt
streamlit run app.py
```

**If port 8501 is busy:**
```bash
streamlit run app.py --server.port 8502
```

**To start (pick one):**
- **Double-click `RUN_APP.vbs`** – Opens a new window (recommended)
- **Double-click `START_APP.bat`** – Opens in current window
- **Terminal:** `.\run.bat` or `streamlit run app.py`

**Connection error?** Keep the CMD window open. Close it = app stops.

## Model

The app looks for a trained model in this order:

1. `tumor_model.h5` (in this folder)
2. `../best_cnn_model.h5` (parent folder)

Place your trained `.h5` model in one of these locations.

## API Key

Set your Groq API key via:

- **Option A:** `.streamlit/secrets.toml` → `GROQ_API_KEY = "your-key"`
- **Option B:** Environment variable `GROQ_API_KEY`

## Error Handling

The app handles:
- Invalid uploaded files
- Image preprocessing failures
- Model loading errors
- Groq API failures

## Project Structure

```
brain_tumor_mri_ai/
├── app.py           # Streamlit UI
├── model.py         # TensorFlow prediction
├── llm.py           # Groq LLM integration
├── prompts.py       # System/user prompts
├── utils.py         # Image preprocessing
├── grad_cam.py      # Grad-CAM visualization
├── requirements.txt
├── run.bat          # Windows run script
├── run.ps1          # PowerShell run script
└── .streamlit/
    ├── config.toml  # Server config
    └── secrets.toml # API key
```
