@echo off
REM Launch the Train Viewer GUI (Windows, torch-free Tkinter app).
setlocal
cd /d "%~dp0"

REM Use the project venv if it exists, otherwise fall back to system python.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" src\TrainViewer\main.py
if errorlevel 1 pause
endlocal
