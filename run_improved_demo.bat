@echo off
echo 🧠 MELVIN IMPROVED WEB SEARCH DEMONSTRATION
echo ===========================================
echo Testing improved web search with clearer responses
echo No Python dependencies - Pure C++ implementation
echo.

echo Compiling melvin_improved_demo.cpp...
g++ -std=c++17 -O2 -o melvin_improved_demo.exe melvin_improved_demo.cpp melvin_optimized_v2.cpp
if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Compilation successful. Running demo...
melvin_improved_demo.exe
pause
