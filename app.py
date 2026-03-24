"""
Brain Tumor MRI AI Assistant - PyTorch Version
Streamlit application for MRI upload, tumor classification, and LLM-assisted Q&A.
Uses PyTorch for model inference with proper CPU optimization.
"""

import json
import os
import io
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from src.llm_cache import fast_chat as llm_chat
from exact_brain_tumor_model import BrainTumorCNN, load_brain_tumor_model

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Brain Tumor MRI AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved UI and confidence bar
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2c5282;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .result-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #bae6fd;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        background: #e2e8f0;
        overflow: hidden;
        margin: 0.2rem 0 1rem 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #0ea5e9, #06b6d4);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .prob-item {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 500;
        color: #334155;
        margin-top: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    div.stSpinner > div {
        text-align: center;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)


# Global model cache
@st.cache_resource
def load_model_cached():
    """Load and cache PyTorch model for better performance."""
    try:
        # Try to load complete model first (with metadata)
        if os.path.exists('brain_tumor_model_complete.pth'):
            model, device, metadata = load_brain_tumor_model('brain_tumor_model_complete.pth', 'complete')
            return model, device, metadata, "✅ Complete model loaded"
        elif os.path.exists('best_brain_tumor_model.pth'):
            model, device, metadata = load_brain_tumor_model('best_brain_tumor_model.pth', 'state_dict')
            # Set default metadata
            metadata = {
                'class_names': ['glioma', 'meningioma', 'notumor', 'pituitary'],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'best_val_accuracy': None
            }
            return model, device, metadata, "✅ State dict model loaded"
        else:
            return None, None, None, "❌ No PyTorch model found. Please place 'brain_tumor_model_complete.pth' in the app directory."
        
    except Exception as e:
        return None, None, None, f"❌ Failed to load PyTorch model: {str(e)}"


def get_groq_api_key() -> Optional[str]:
    """Get Groq API key from Streamlit secrets or environment."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
        
    return os.environ.get("GROQ_API_KEY")


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load image from bytes using PIL."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def preprocess_image(image: Image.Image, metadata: Dict[str, Any]) -> torch.Tensor:
    """Preprocess image for PyTorch model inference."""
    mean = metadata.get('mean', [0.485, 0.456, 0.406])
    std = metadata.get('std', [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


@st.cache_data(show_spinner=False)
def get_prediction(image_bytes: bytes, _model: torch.nn.Module, _device: torch.device, _metadata: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
    """Cached prediction function so we don't re-run inference on every interaction."""
    # Load and preprocess
    image = load_image_from_bytes(image_bytes)
    image_tensor = preprocess_image(image, _metadata).to(_device)
    
    class_names = _metadata.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
    
    # Run inference
    _model.eval()
    with torch.no_grad():
        outputs = _model(image_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
    # Get max and map all probabilities
    confidence_score = float(max(probabilities))
    predicted_idx = int(probabilities.argmax())
    predicted_class = class_names[predicted_idx]
    
    # Create dictionary of all class probabilities
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    # Sort descending
    prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
    
    return predicted_class, confidence_score, prob_dict


def reset_app():
    """Clear session state to reset the app."""
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    st.session_state.uploader_key += 1


def build_sidebar():
    """Constructs the sidebar interface."""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3209/3209041.png", width=60)
        st.title("Settings & Info")
        
        # We removed the manual API configuration section
        if not get_groq_api_key():
            st.warning("Chat Assistant requires a Groq API key set in code/environment.")

        st.markdown("---")
        st.markdown("### 🧠 Model Specifications")
        
        # Display model load status
        if st.session_state.get('model_status'):
            st.caption(f"Status: {st.session_state.model_status}")
            
        metadata = st.session_state.get('model_metadata', {})
        val_acc = metadata.get('best_val_accuracy')
        if val_acc:
            st.metric("Model Fidelity", f"{val_acc:.2f}% validation")
        else:
            st.metric("Model Focus", "4 Brain Tumor Classes")
            
        st.caption(f"Classes: {', '.join([c.title() for c in metadata.get('class_names', [])])}")

        st.markdown("---")
        st.markdown("### 🛠️ Actions")
        if st.button("🔄 Clear Data & Reset App"):
            reset_app()
            st.rerun()
            
        st.markdown("---")
        st.markdown("""
        **Disclaimer:**
        *This tool is designed for educational purposes and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician.*
        """)


def main():
    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    
    # Load model
    if "model" not in st.session_state:
        with st.spinner("🔄 Initializing Neural Network..."):
            model, device, metadata, status = load_model_cached()
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_metadata = metadata or {}
            st.session_state.model_status = status
            
    # Sidebar rendering
    build_sidebar()
            
    # Main UI
    st.markdown('<h1 class="main-header">Brain Tumor MRI AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an MRI scan to perform automated neural network analysis and receive an interactive clinical explanation.</p>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.error(st.session_state.model_status)
        return

    # --- Upload Area ---
    # We use uploader_key to force it to clear when reset is configured
    uploaded_file = st.file_uploader(
        "Upload Brain MRI (.JPG, .PNG)",
        type=["jpg", "jpeg", "png"],
        help="Ensure the MRI is clear and centered.",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if not uploaded_file:
        st.info("👆 Please upload an MRI image to begin analysis.")
        
        # Optional placeholder aesthetic display when empty
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/2864/2864320.png", use_container_width=True)
        return

    # Read bytes safely
    try:
        image_bytes = uploaded_file.read()
        if len(image_bytes) == 0:
            st.error("❌ The uploaded file is empty.")
            return
            
        # Displayable image
        display_image = load_image_from_bytes(image_bytes)
    except Exception as e:
        st.error(f"❌ Invalid file: {str(e)}")
        return

    st.markdown("---")

    # --- Dashboard View (Image + Results) ---
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown('<p class="section-header">🖼️ MRI Scan Preview</p>', unsafe_allow_html=True)
        st.image(display_image, use_container_width=True, caption=uploaded_file.name)
        
    with col2:
        st.markdown('<p class="section-header">🔬 Analysis Results</p>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing neural patterns..."):
            try:
                predicted_class, top_confidence, all_probs = get_prediction(
                    image_bytes, 
                    st.session_state.model, 
                    st.session_state.device, 
                    st.session_state.model_metadata
                )
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                return
                
        # Main Prediction Box
        display_name = predicted_class.replace('_', ' ').title()
        fill_color = "#10b981" if display_name == "Notumor" else "#ef4444"
        
        st.markdown(f"""
        <div class="result-box">
            <p style="margin:0; color:#475569; font-weight:600;">Primary Detection:</p>
            <p style="font-size: 2rem; font-weight:700; color: {fill_color}; margin: 0.2rem 0;">{display_name}</p>
            <div style="margin-top: 1rem;">
                <p style="margin:0; color:#475569; font-weight:600;">Overall Confidence: {top_confidence:.1%}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {min(100, top_confidence * 100)}%; background: {fill_color};"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Probabilities
        with st.expander("📊 View Detailed Probabilities", expanded=True):
            for cls_name, prob in all_probs.items():
                cls_display = cls_name.replace('_', ' ').title()
                st.markdown(f"""
                <div class="prob-item">
                    <span>{cls_display}</span>
                    <span>{prob:.2%}</span>
                </div>
                <div class="confidence-bar" style="height: 6px; margin-bottom: 0.8rem;">
                    <div class="confidence-fill" style="width: {min(100, prob * 100)}%;"></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Chat Assistant View ---
    st.markdown('<p class="section-header">💬 Clinical Assistant Q&A</p>', unsafe_allow_html=True)
    st.caption(f"The LLM is aware that this scan was predicted as **{display_name}**. Ask questions about this specific diagnosis or tumor properties.")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    api_key_status = get_groq_api_key()
    
    if user_question := st.chat_input("E.g., What are the common symptoms for this type of tumor?", disabled=(not api_key_status)):
        if not api_key_status:
            st.error("Please add a Groq API Key in the sidebar to use the Assistant.")
            return

        # Show user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating clinical insight..."):
                try:
                    response = llm_chat(
                        predicted_class=predicted_class,
                        confidence=top_confidence,
                        question=user_question,
                        api_key=api_key_status
                    )
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Groq API error: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected critical error occurred.")
        st.exception(e)
