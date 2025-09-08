@echo off
echo 🧠 MELVIN COMPLETE SYSTEM TEST
echo ==============================
echo Testing Melvin's unified brain with all systems
echo.

echo Available Melvin executables:
echo ==============================
dir *.exe | findstr melvin

echo.
echo Choose which version to run:
echo 1. Simple Demo (melvin_simple.exe)
echo 2. Full Demo (melvin_demo.exe) 
echo 3. Interactive Batch Mode (run_melvin_interactive.bat)
echo 4. Dynamic Brain (melvin_dynamic.exe)
echo 5. Unified Brain (melvin_unified.exe)
echo 6. Natural Language (melvin_natural.exe)
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo Running Simple Demo...
    melvin_simple.exe
) else if "%choice%"=="2" (
    echo Running Full Demo...
    melvin_demo.exe
) else if "%choice%"=="3" (
    echo Running Interactive Batch Mode...
    run_melvin_interactive.bat
) else if "%choice%"=="4" (
    echo Running Dynamic Brain...
    melvin_dynamic.exe
) else if "%choice%"=="5" (
    echo Running Unified Brain...
    melvin_unified.exe
) else if "%choice%"=="6" (
    echo Running Natural Language...
    melvin_natural.exe
) else (
    echo Invalid choice. Running default demo...
    melvin_demo.exe
)

echo.
echo Test completed!
pause
