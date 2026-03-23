@echo off
cd /d "%~dp0"
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
echo.
echo ========================================
echo  Brain Tumor MRI AI Assistant
echo ========================================
echo.
echo Starting Streamlit... Open http://localhost:8501
echo KEEP THIS WINDOW OPEN - closing it stops the app
echo.
streamlit run app.py --server.port 8501
pause
