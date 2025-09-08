@echo off
echo 🧠 Building Simplified Upgraded Melvin Unified Brain System
echo ==========================================================

REM Set up compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Compiling simplified upgraded unified brain system...
echo.

REM Compile with minimal dependencies (no external JSON library)
g++ -std=c++17 -O2 -Wall -Wextra ^
    -I. ^
    -I"C:\msys64\mingw64\include" ^
    -I"C:\msys64\mingw64\include\curl" ^
    melvin_unified_brain.cpp ^
    melvin_unified_interactive.cpp ^
    -o melvin_simple_upgraded.exe ^
    -lcurl ^
    -lz ^
    -lpthread ^
    -static-libgcc ^
    -static-libstdc++

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Build successful! Created melvin_simple_upgraded.exe
    echo.
    echo 🧠 SIMPLIFIED UPGRADED MELVIN FEATURES:
    echo ======================================
    echo ✅ Background scheduler (autonomous thinking every 30s)
    echo ✅ Ollama integration (HTTP API queries, simple JSON parsing)
    echo ✅ Force-driven responses (0.0-1.0 continuous values)
    echo ✅ Contradiction detection and regeneration
    echo ✅ BinaryNode storage for ALL I/O (user, self, Ollama)
    echo ✅ Instinct pressures guide both user and autonomous thought
    echo ✅ Hebbian learning with question-answer connections
    echo ✅ Transparent reasoning paths with confidence scores
    echo ✅ Simple string-based JSON parsing (no external libraries)
    echo.
    echo 🚀 Ready to run with: melvin_simple_upgraded.exe
    echo.
    echo 📋 REQUIREMENTS:
    echo - Ollama running on localhost:11434 (optional)
    echo - BING_API_KEY environment variable (optional)
    echo - Internet connection for web search (optional)
    echo - MinGW-w64 compiler with libcurl and zlib
) else (
    echo.
    echo ❌ Build failed! Check for missing dependencies:
    echo - MinGW-w64 compiler
    echo - libcurl development libraries
    echo - zlib development libraries
    echo - pthread library
)

echo.
pause
