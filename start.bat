@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found. See README to install dependencies.
  pause
  exit /b 1
)
.venv\Scripts\python.exe run.py --open
