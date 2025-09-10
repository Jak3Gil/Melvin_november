@echo off
echo 🧠 CHECKING MELVIN'S NEURAL NETWORK STATUS
echo ==========================================

REM Set up MinGW compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling status checker...
g++ -std=c++17 -o check_status.exe check_melvin_status.cpp

if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Running status check...
check_status.exe

echo.
echo To get live counts, run: run_unified_melvin.bat
pause
