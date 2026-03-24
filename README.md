# 🧠 Brain Tumor MRI AI Assistant

An advanced, AI-powered web application designed to analyze Brain MRI scans, classify tumors using PyTorch Convolutional Neural Networks, and provide interactive clinical explanations via a specialized Groq-powered LLM assistant.

## ✨ Features
- **🖼️ MRI Image Analysis** – Upload `.jpg`, `.jpeg` or `.png` scans for instant processing.
- **🔬 Tumor Classification** – Uses a custom PyTorch model to detect 4 classes: Glioma, Meningioma, Pituitary, or No Tumor.
- **📊 Confidence Visualizer** – Provides detailed probability bars for primary and secondary detections.
- **💬 Clinical LLM Chat** – An integrated clinical assistant powered by Groq (Llama 3) that understands the MRI context and answers related diagnosis questions.
- **⚡ Performance Optimized** – Features dedicated prediction workers and cached model weights for faster throughput.

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Mshabrawy-m/BrainTumor.git
cd BrainTumor
pip install -r requirements.txt
```

### 2. API Key Configuration
To enable the Clinical Chat Assistant, you need a free Groq API key:
- Set it as an environment variable: `GROQ_API_KEY="your_api_key"`
- Or create a `.streamlit/secrets.toml` file with:
  ```toml
  GROQ_API_KEY = "your_api_key"
  ```

### 3. Run the Application
You can start the application using one of the provided scripts in the `scripts/` folder:
- **Windows Shortcut** (Recommended): Double-click `scripts/RUN_APP.vbs` to start the server in the background and open your browser automatically.
- **Batch Script**: Double-click `scripts/START_APP.bat` to run the active server in a terminal.
- **Command Line**: Run `streamlit run app.py` anywhere in the root directory.

## 📂 Project Structure
```text
BrainTumor/
├── models/
│   ├── best_brain_tumor_model.pth        # Optimized state_dict PyTorch model
│   └── brain_tumor_model_complete.pth    # Complete PyTorch model with metadata
├── scripts/
│   ├── RUN_APP.vbs                       # Background Windows starter
│   ├── START_APP.bat                     # Foreground batch starter
│   └── run.bat                           # Alternative batch starter
├── src/
│   ├── exact_brain_tumor_model.py        # PyTorch Neural Network architecture
│   ├── llm_cache.py                      # Groq API integration and prompt logic
│   ├── predict_worker.py                 # Multi-processing prediction worker
│   └── utils/
│       └── utils.py                      # Shared application utilities
├── app.py                                # Main Streamlit Web Interface
├── Dockerfile                            # Docker container configuration
├── DEPLOYMENT.md                         # Deployment guide and instructions
└── requirements.txt                      # Project dependencies
```

## ⚠️ Disclaimer
*This tool is designed for educational and research purposes and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified physician.*
