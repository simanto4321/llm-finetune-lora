@echo off
call "%~dp0..\env_g_drive.bat"
set "VENV_PY=%~dp0..\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Run G:\portfolio\setup_all.bat first.
    exit /b 1
)

"%VENV_PY%" "%~dp0train.py" %*
