@echo off
echo 🧠 Building Melvin Unified Brain with Node-Based Memory System
echo =============================================================

REM Set up compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

REM Compile the unified version
echo Compiling unified Melvin brain system...
g++ -std=c++17 -I. -I./src src/main_unified.cpp src/logging.cpp src/websearch_nodes.cpp src/memory_nodes.cpp -o melvin_unified.exe -lcurl

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Compilation successful!
    echo Created: melvin_unified.exe
    echo.
    echo Testing the unified brain system...
    echo.
    echo Running diagnostics...
    melvin_unified.exe --diag
    echo.
    echo 🎉 Melvin Unified Brain System is ready!
    echo.
    echo Features:
    echo - Node-based memory system
    echo - Wikipedia and DuckDuckGo search
    echo - Global memory storage
    echo - Thread-safe logging
    echo.
    echo You can now run: melvin_unified.exe
    echo Or use: run_unified_melvin.bat
) else (
    echo.
    echo ❌ Compilation failed!
    echo Check for errors above.
)

echo.
pause
