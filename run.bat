@echo off
REM Hiring-RH Talent Orbit: one-click start on Windows.
cd /d "%~dp0"
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv || (echo Python 3.10+ is required & pause & exit /b 1)
)
call venv\Scripts\activate.bat
echo Installing requirements (first run takes a few minutes)...
python -m pip install -q -r requirements.txt || (pause & exit /b 1)
if not exist .env (
  copy .env.example .env >nul
  echo Created .env from .env.example. Edit it to choose your LLM.
)
echo.
echo Talent Orbit: http://localhost:8000   (Ctrl+C to stop)
start "" http://localhost:8000
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
