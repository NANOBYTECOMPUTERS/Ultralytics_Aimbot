@echo off
echo Checking for Python and required packages...

:: Check for Python Installation
python --version >NUL 2>&1 
if %errorlevel% neq 0 (
    echo Python is not installed. Please download and install Python from https://www.python.org/downloads/
    pause
    exit /b 1 
)

:: Check and Install Pip if Needed
python -m pip --version >NUL 2>&1
if %errorlevel% neq 0 (
    echo Installing pip...
    python -m ensurepip --upgrade
    if %errorlevel% neq 0 (
        echo Failed to install pip. Please try installing manually.
        pause
        exit /b 1
    )
)

:: Check and Install Streamlit if Needed
python -c "import streamlit" 2>NUL
if %errorlevel% neq 0 (
    echo Installing Streamlit...
    pip install streamlit
    if %errorlevel% neq 0 (
        echo Failed to install Streamlit. Please check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo Starting your Streamlit app...
streamlit run helper.py
