@echo off
call "%~dp0..\env_g_drive.bat"
set "VENV_PY=%~dp0..\.venv\Scripts\python.exe"
"%VENV_PY%" "%~dp0inference.py" %*
