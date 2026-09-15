@echo off
echo Starting SHI Tail Gate application...
start /b python -c "import time, webbrowser; time.sleep(2); webbrowser.open('http://localhost:5000')"
python app.py
pause
