# 🚀 DEPLOYMENT GUIDE

## Streamlit Cloud Deployment

### Step 1: Prepare Repository
```bash
# Ensure all changes are committed
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: GitHub Release for Model
```bash
# Create GitHub release and upload model
gh release create v1.0 data/best_cnn_model.h5 --title "Model Files" --notes "Pre-trained CNN model for brain tumor classification"
```

### Step 3: Streamlit Cloud Setup
1. Go to [streamlit.io](https://streamlit.io)
2. Click "New app" → "Deploy from GitHub"
3. Select your repository
4. Configure:
   - **Main file path**: `app.py`
   - **Python version**: `3.11`
   - **Dependencies**: `requirements.txt`

### Step 4: Environment Variables
Set these in Streamlit Cloud secrets:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
MODEL_URL = "https://github.com/yourusername/brain-tumor-mri-ai/releases/download/v1.0/best_cnn_model.h5"
```

### Step 5: Deploy
- Click "Deploy"
- Monitor build logs
- Test the deployed app

## Docker Deployment

### Local Docker
```bash
# Build image
docker build -t brain-tumor-mri-ai .

# Run container
docker run -p 8501:8501 -e GROQ_API_KEY=your_key brain-tumor-mri-ai
```

### Docker with Model URL
```bash
# Build with model download
docker build --build-arg MODEL_URL=https://github.com/yourusername/brain-tumor-mri-ai/releases/download/v1.0/best_cnn_model.h5 -t brain-tumor-mri-ai .

# Run
docker run -p 8501:8501 -e GROQ_API_KEY=your_key brain-tumor-mri-ai
```

## Production Checklist

### ✅ Pre-Deployment
- [ ] All tests passing
- [ ] Model file uploaded to GitHub releases
- [ ] Environment variables configured
- [ ] Requirements.txt pinned to exact versions
- [ ] .gitignore excludes sensitive files

### ✅ Post-Deployment
- [ ] App loads successfully
- [ ] Model prediction works
- [ ] LLM integration functional
- [ ] Error handling tested
- [ ] Performance acceptable

## Troubleshooting

### Common Issues
1. **Model not found**: Check MODEL_URL environment variable
2. **LLM errors**: Verify GROQ_API_KEY is valid
3. **Slow loading**: Model caching needs first load
4. **Memory issues**: Increase Streamlit Cloud memory tier

### Debug Mode
Set `DEBUG=true` environment variable for detailed logs.
