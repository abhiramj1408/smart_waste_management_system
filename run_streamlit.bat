@echo off
setlocal
cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
    echo [INFO] Using parent virtual environment...
    "..\.venv\Scripts\python.exe" -m streamlit run main.py
) else if exist ".venv\Scripts\python.exe" (
    echo [INFO] Using local virtual environment...
    ".venv\Scripts\python.exe" -m streamlit run main.py
) else (
    echo [INFO] Using system python...
    python -m streamlit run main.py
)

pause
