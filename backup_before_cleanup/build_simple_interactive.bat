@echo off
echo 🧠 MELVIN SIMPLE INTERACTIVE MODE
echo =================================
echo Building Melvin's simplified interactive system
echo (No external dependencies required)
echo.

REM Set compiler flags
set CXXFLAGS=-O3 -std=c++17 -Wall -Wextra

echo Compiling simplified interactive Melvin...
g++ %CXXFLAGS% -o melvin_simple_interactive.exe melvin_simple_interactive.cpp melvin_optimized_v2.cpp -lz -llzma -lzstd

if %ERRORLEVEL% EQU 0 (
    echo ✅ Compilation successful! 
    echo.
    echo Starting interactive session...
    echo ================================
    melvin_simple_interactive.exe
) else (
    echo ❌ Compilation failed!
    echo.
    echo Trying alternative compilation without compression libraries...
    g++ %CXXFLAGS% -o melvin_simple_interactive.exe melvin_simple_interactive.cpp melvin_optimized_v2.cpp
    
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Alternative compilation successful! 
        echo.
        echo Starting interactive session...
        echo ================================
        melvin_simple_interactive.exe
    ) else (
        echo ❌ Compilation failed!
        echo.
        echo Common issues:
        echo - Make sure you have g++ installed
        echo - Check that melvin_optimized_v2.h and melvin_optimized_v2.cpp exist
        echo - Try installing compression libraries: zlib, lzma, zstd
    )
)

echo.
echo Session ended.
pause
