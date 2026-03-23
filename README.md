# 🧠 Brain Tumor MRI AI Assistant

A cutting-edge AI-powered medical imaging application for brain tumor MRI analysis. Combines deep learning classification with intelligent conversational AI to assist healthcare professionals in rapid tumor detection and clinical decision support.

## 🎯 Mission

To provide accurate, accessible, and intelligent brain tumor detection from MRI scans using state-of-the-art deep learning models, enhanced with contextual AI assistance for clinical queries and educational support.

## ⚡ Key Features

### 🏥 **Medical Imaging Analysis**
- **📸 Multi-format Support** – JPG/PNG MRI image upload with drag-and-drop interface
- **🧬 Advanced Classification** – PyTorch CNN model predicts 4 classes: glioma, meningioma, pituitary, no tumor
- **📊 Confidence Visualization** – Interactive confidence bars with percentage scores
- **🔍 Explainable AI** – Grad-CAM heatmaps highlighting tumor regions for interpretability

### 🤖 **Intelligent Clinical Assistant**
- **💬 Context-Aware Chat** – LLM-powered assistant aware of current diagnosis
- **🧠 Medical Knowledge Base** – Specialized in brain tumor MRI interpretation
- **📝 Conversation Memory** – Persistent chat history for continuity
- **⚡ Real-time Responses** – Fast inference with model caching

### 🛠️ **Technical Excellence**
- **⚡ High Performance** – PyTorch optimization with CPU acceleration
- **🔄 Model Caching** – Single model load for server stability
- **🌐 Responsive Design** – Modern UI with TailwindCSS styling
- **🔧 Port Flexibility** – Automatic port fallback (8501 → 8502)

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

## 🧬 Model Architecture

### **Deep Learning Model**
- **Framework**: PyTorch CNN with optimized architecture
- **Input**: 224x224 RGB MRI images
- **Classes**: 4 tumor types (glioma, meningioma, pituitary, no tumor)
- **Model Files**:
  - `brain_tumor_model_complete.pth` (primary model with metadata)
  - `best_brain_tumor_model.pth` (backup model)

### **Model Loading Priority**
1. `brain_tumor_model_complete.pth` (with metadata)
2. `best_brain_tumor_model.pth` (fallback)

The system automatically detects and loads the best available model with performance optimization.

## API Key

Set your Groq API key via:

- **Option A:** `.streamlit/secrets.toml` → `GROQ_API_KEY = "your-key"`
- **Option B:** Environment variable `GROQ_API_KEY`

## 🔒 Error Handling & Reliability

The application includes comprehensive error handling for:
- **📁 File Validation** – Invalid format/size detection
- **🖼️ Image Processing** – Preprocessing failure recovery
- **🧠 Model Loading** – Graceful fallback mechanisms
- **🌐 API Integration** – Network error handling with retry logic
- **💾 Memory Management** – Resource cleanup and optimization

## 📁 Project Structure

```
brain_tumor_ai-1/
├── 📄 app.py                    # Main Streamlit application
├── 🧠 exact_brain_tumor_model.py # PyTorch model architecture
├── 💬 src/
│   ├── llm_cache.py            # LLM integration with caching
│   ├── performance.py          # Model optimization utilities
│   └── model.py               # Model loading utilities
├── 📋 requirements.txt         # Python dependencies
├── 🚀 predict_worker.py        # Background prediction worker
├── 🖼️ app_minimal.py          # Minimal version for testing
├── 📊 brain_tumor_model_complete.pth  # Primary trained model
├── 📊 best_brain_tumor_model.pth      # Backup model
├── 🔧 .streamlit/
│   ├── config.toml            # Streamlit configuration
│   └── secrets.toml           # API keys (secure)
└── 📖 README.md               # This documentation
```

## 🏥 Clinical Applications

### **Use Cases**
- **🔍 Rapid Screening** – Initial tumor detection in clinical settings
- **📚 Medical Education** – Teaching tool for radiology students
- **🏥 Remote Consultation** – Telemedicine support for rural areas
- **📊 Research Support** – Data analysis for medical research

### **Tumor Types Detected**
1. **🧬 Glioma** – Primary brain tumors from glial cells
2. **🦠 Meningioma** – Tumors from meninges membranes
3. **🎯 Pituitary** – Pituitary gland abnormalities
4. **✅ No Tumor** – Normal brain scans

## ⚠️ Medical Disclaimer

> **Important**: This tool is designed for **assistive purposes only** and should not replace professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions and treatment planning.

## 🤝 Contributing

We welcome contributions to improve the accuracy and functionality of this medical AI tool. Please ensure all contributions follow medical AI ethics and data privacy standards.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
