@echo off
echo Running Claude Automation - Code Tasks Session
cd /d "%~dp0"
python -m src.main --session session2 --debug
echo Session completed
pause