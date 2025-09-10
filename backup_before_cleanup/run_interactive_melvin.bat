@echo off
echo 🧠 MELVIN INTERACTIVE MODE
echo =========================
echo Starting Melvin's interactive conversation system
echo.

REM Set up MinGW compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling interactive Melvin...
g++ -std=c++17 -O2 -o melvin_interactive.exe melvin_interactive.cpp

if %errorlevel% neq 0 (
    echo Compilation failed.
    echo Trying alternative compilation...
    g++ -std=c++17 -o melvin_interactive.exe melvin_interactive.cpp
    if %errorlevel% neq 0 (
        echo Compilation failed. Check for errors above.
        pause
        exit /b %errorlevel%
    )
)

echo ✅ Compilation successful!
echo.
echo Starting interactive session...
echo ================================
melvin_interactive.exe

echo.
echo Session ended.
pause
