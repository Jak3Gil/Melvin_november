@echo off
echo 🧠 MELVIN ENHANCED INTERACTIVE MODE
echo ===================================
echo Starting Melvin with enhanced personality and knowledge
echo.

REM Set up MinGW compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling enhanced interactive Melvin...
g++ -std=c++17 -O2 -o melvin_improved_interactive.exe melvin_improved_interactive.cpp

if %errorlevel% neq 0 (
    echo Compilation failed.
    echo Trying alternative compilation...
    g++ -std=c++17 -o melvin_improved_interactive.exe melvin_improved_interactive.cpp
    if %errorlevel% neq 0 (
        echo Compilation failed. Check for errors above.
        pause
        exit /b %errorlevel%
    )
)

echo ✅ Compilation successful!
echo.
echo Starting enhanced interactive session...
echo ========================================
melvin_improved_interactive.exe

echo.
echo Session ended.
pause
