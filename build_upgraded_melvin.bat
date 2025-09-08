@echo off
echo 🧠 Building Upgraded Melvin Unified Brain System
echo ===============================================

REM Set up compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling upgraded unified brain system...
echo.

REM Compile with all necessary libraries
g++ -std=c++17 -O2 -Wall -Wextra ^
    -I. ^
    -I"C:\msys64\mingw64\include" ^
    -I"C:\msys64\mingw64\include\curl" ^
    -I"C:\msys64\mingw64\include\nlohmann" ^
    melvin_unified_brain.cpp ^
    melvin_unified_interactive.cpp ^
    -o melvin_upgraded.exe ^
    -lcurl ^
    -lz ^
    -lpthread ^
    -static-libgcc ^
    -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Build successful! Created melvin_upgraded.exe
    echo.
    echo 🧠 UPGRADED MELVIN UNIFIED BRAIN FEATURES:
    echo ==========================================
    echo ✅ Background scheduler (autonomous thinking every 30s)
    echo ✅ Ollama integration (HTTP API queries)
    echo ✅ Force-driven responses (0.0-1.0 continuous values)
    echo ✅ Contradiction detection and regeneration
    echo ✅ BinaryNode storage for ALL I/O (user, self, Ollama)
    echo ✅ Instinct pressures guide both user and autonomous thought
    echo ✅ Hebbian learning with question-answer connections
    echo ✅ Transparent reasoning paths with confidence scores
    echo.
    echo 🚀 Ready to run with: melvin_upgraded.exe
    echo.
    echo 📋 REQUIREMENTS:
    echo - Ollama running on localhost:11434 (optional)
    echo - BING_API_KEY environment variable (optional)
    echo - Internet connection for web search (optional)
) else (
    echo.
    echo ❌ Build failed! Check for missing dependencies:
    echo - MinGW-w64 compiler
    echo - libcurl development libraries
    echo - zlib development libraries
    echo - nlohmann/json header
    echo - pthread library
)

echo.
pause
