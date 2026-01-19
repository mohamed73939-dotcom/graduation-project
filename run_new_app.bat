@echo off
TITLE Sidecut AI Launcher (Modern UI)
echo ==================================================
echo       Sidecut ^| AI Video Summarizer (NiceGUI)
echo ==================================================
echo.

echo [1/2] Launching Backend Server...
:: Navigate to backend and start uvicorn
start "Sidecut Backend" cmd /k "cd backend && python -m uvicorn api:app --host 0.0.0.0 --port 8000"

echo.
echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo.
start "Sidecut Modern UI" cmd /k "python frontend/nice-ui/main.py"

echo.
echo ==================================================
echo    Application is running!
echo    - UI: http://localhost:8080
echo    - API: http://localhost:8000/docs
echo ==================================================
echo.
echo Please close any previous Sidecut windows before running this to avoid port conflicts.
pause
