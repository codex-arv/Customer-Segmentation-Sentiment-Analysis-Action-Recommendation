@echo off
echo Stopping FastAPI server on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /PID %%a /F
    echo Killed PID %%a
)
echo Done.