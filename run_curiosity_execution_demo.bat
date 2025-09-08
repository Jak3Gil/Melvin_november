@echo off
echo 🧠 MELVIN CURIOSITY EXECUTION LOOP DEMONSTRATION
echo ===============================================
echo Testing Phase 6.8 - Curiosity Execution Loop
echo Features:
echo - Separation of internal/external channels
echo - Curiosity execution flow (recall → tools → meta-tools)
echo - Non-repetitive, evidence-backed responses
echo - Moral safety filtering
echo.

echo Compiling melvin_curiosity_execution_demo.cpp...
g++ -std=c++17 -O2 -o melvin_curiosity_execution_demo.exe melvin_curiosity_execution_demo.cpp melvin_optimized_v2.cpp
if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Compilation successful. Running demo...
melvin_curiosity_execution_demo.exe
pause
