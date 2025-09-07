@echo off
echo 🧠 MELVIN SIMPLE WORKING DEMONSTRATION
echo ======================================
echo Testing core reasoning with clearer responses
echo No complex dependencies - Pure C++ implementation
echo.

REM Set up MinGW compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling melvin_simple_working.cpp...
g++ -std=c++17 -O2 -o melvin_simple_working.exe melvin_simple_working.cpp

if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Compilation successful. Running demo...
echo.
melvin_simple_working.exe
echo.
pause
