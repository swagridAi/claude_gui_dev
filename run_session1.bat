@echo off
echo Running Claude Automation - Research Questions Session
cd /d "%~dp0"
python -m src.main --session session1 --debug
echo Session completed
pause