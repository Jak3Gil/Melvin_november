@echo off
echo 🧠 MELVIN DYNAMIC INTERACTIVE SYSTEM
echo ====================================
echo Starting Melvin with dynamic personality, context weaving, and curiosity injection
echo.

REM Set up MinGW compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling dynamic Melvin...
g++ -std=c++17 -O2 -o melvin_dynamic.exe melvin_dynamic_interactive.cpp

if %errorlevel% neq 0 (
    echo Compilation failed.
    echo Trying alternative compilation...
    g++ -std=c++17 -o melvin_dynamic.exe melvin_dynamic_interactive.cpp
    if %errorlevel% neq 0 (
        echo Compilation failed. Check for errors above.
        pause
        exit /b %errorlevel%
    )
)

echo ✅ Compilation successful!
echo.
echo Starting dynamic interactive session...
echo =======================================
echo Features active:
echo - 20+ response variants per intent
echo - Context weaving from conversation history
echo - Curiosity-driven questions
echo - Dynamic personality shifts
echo - Anti-repetition algorithms
echo =======================================
melvin_dynamic.exe

echo.
echo Session ended.
pause
