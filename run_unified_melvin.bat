@echo off
echo 🧠 MELVIN UNIFIED DYNAMIC NEURAL SYSTEM
echo ========================================
echo Starting Melvin with actual node creation and connections
echo.

REM Set up MinGW compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling unified neural Melvin...
g++ -std=c++17 -O2 -o melvin_unified.exe melvin_unified_dynamic.cpp

if %errorlevel% neq 0 (
    echo Compilation failed.
    echo Trying alternative compilation...
    g++ -std=c++17 -o melvin_unified.exe melvin_unified_dynamic.cpp
    if %errorlevel% neq 0 (
        echo Compilation failed. Check for errors above.
        pause
        exit /b %errorlevel%
    )
)

echo ✅ Compilation successful!
echo.
echo Starting unified neural interactive session...
echo ==============================================
echo Features active:
echo - Real neural node creation from user input
echo - Connection formation between concepts
echo - Dynamic personality with context awareness
echo - Memory persistence across sessions
echo - Neural network traversal and activation
echo ==============================================
melvin_unified.exe

echo.
echo Session ended. Neural memory saved.
pause
