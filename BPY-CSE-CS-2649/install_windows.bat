@echo off
title FraudShield GAN Setup
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy /Y .env.example .env
python generate_dataset.py
echo.
echo Setup complete. Edit .env if your MySQL root account has a password.
echo Then run: run_windows.bat
pause

