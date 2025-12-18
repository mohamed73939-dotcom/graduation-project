@echo off
TITLE Sidecut AI Launcher
echo ==================================================
echo       Sidecut ^| AI Video Summarizer Launcher
echo ==================================================
echo.

echo [1/2] Launching Backend Server...
:: Navigate to backend and start uvicorn
start "Sidecut Backend" cmd /k "cd backend && python -m uvicorn api:app --host 0.0.0.0 --port 8000"

echo.
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo [2/2] Launching Frontend Interface...
:: Start Streamlit from project root
start "Sidecut Frontend" cmd /k "python -m streamlit run frontend/app.py"

echo.
echo ==================================================
echo    Application is running!
echo    - Frontend: http://localhost:8501
echo    - Backend:  http://localhost:8000/docs
echo ==================================================
echo.
echo You can close this window, but keep the other two command windows open.
pause
