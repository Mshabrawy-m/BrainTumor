@echo off
cd /d "%~dp0\.."
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
set CUDA_VISIBLE_DEVICES=
echo.
echo ========================================
echo  Brain Tumor MRI AI Assistant
echo ========================================
echo.
echo Starting... Open http://localhost:8501
echo.
echo DO NOT CLOSE THIS WINDOW
echo ========================================
echo.
streamlit run app.py --server.port 8501 --server.address localhost
pause
