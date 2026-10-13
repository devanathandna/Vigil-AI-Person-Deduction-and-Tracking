@echo off
REM Run from the project folder. Requires Python 3.10 - 3.12 on PATH.
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :error
pip install -r requirements.txt || goto :error
echo.
echo Setup complete. Next time:  .venv\Scripts\activate  then  python main.py
echo (First run downloads yolov8n.pt, so internet is needed once.)
exit /b 0
:error
echo Setup failed - see messages above.
exit /b 1
