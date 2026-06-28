@echo off
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Starting the Fraud Detection System...
echo The application will open in your default web browser shortly.
python -m streamlit run app.py
pause
