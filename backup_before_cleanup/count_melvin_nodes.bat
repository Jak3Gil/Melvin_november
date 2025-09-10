@echo off
echo 🧠 COUNTING MELVIN'S NODES AND CONNECTIONS
echo ==========================================

REM Set up MinGW compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling binary reader...
g++ -std=c++17 -o melvin_binary_reader.exe melvin_binary_reader.cpp

if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Running binary reader...
melvin_binary_reader.exe

echo.
echo To see live node creation, run: run_unified_melvin.bat
pause
