' Double-click to start Brain Tumor MRI AI Assistant in a new window
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = appDir & "\.."
WshShell.Run "cmd /k ""title Brain Tumor MRI AI && cd /d """ & appDir & "\.."" && set TF_ENABLE_ONEDNN_OPTS=0 && set TF_CPP_MIN_LOG_LEVEL=3 && echo. && echo Open http://localhost:8501 in your browser && echo Keep this window open! && echo. && streamlit run app.py --server.port 8501 --server.address localhost""", 1, False
