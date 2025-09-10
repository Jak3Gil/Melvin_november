@echo off
echo 🧠 MELVIN C++ INTERACTIVE SYSTEM
echo =================================
echo Welcome! I'm Melvin, your unified brain AI companion.
echo I have integrated systems for:
echo - Curiosity Gap Detection
echo - Dynamic Tools System
echo - Meta-Tool Engineer
echo - Curiosity Execution Loop
echo - Temporal Planning & Sequencing
echo - Enhanced Web Search (Pure C++)
echo.
echo Type 'quit' to exit, 'status' for system info, 'help' for commands.
echo =================================

echo Compiling melvin_cpp_interactive.cpp...
g++ -std=c++17 -O2 -o melvin_cpp_interactive.exe melvin_cpp_interactive.cpp melvin_optimized_v2.cpp
if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Compilation successful. Running interactive system...
melvin_cpp_interactive.exe
pause
