@echo off
echo 🧠 Starting Melvin Unified Brain System
echo ======================================

REM Set up compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

if not exist melvin_unified.exe (
    echo ❌ melvin_unified.exe not found!
    echo Please run build_unified_melvin.bat first.
    pause
    exit /b 1
)

echo Starting Melvin Unified Brain...
echo.
echo This system converts all inputs into nodes and connections,
echo stores them in global memory, and can search Wikipedia/DuckDuckGo.
echo.
echo Commands:
echo - Type anything to process and store in memory
echo - Type 'memory' to see memory status
echo - Type 'diag' for diagnostics
echo - Type 'quit' to exit
echo.

melvin_unified.exe

echo.
echo Melvin Unified Brain session ended.
pause