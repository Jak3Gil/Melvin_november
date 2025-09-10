@echo off
echo Building Melvin Instinct System Test...

REM Set compiler flags for optimization
set CXXFLAGS=-O3 -march=native -ffast-math -std=c++17 -Wall -Wextra

REM Compile the instinct system test
g++ %CXXFLAGS% -o test_instinct_system.exe test_instinct_system.cpp melvin_optimized_v2.cpp -lz -llzma -lzstd

if %ERRORLEVEL% EQU 0 (
    echo ✅ Build successful! 
    echo.
    echo Running instinct system test...
    echo ================================
    test_instinct_system.exe
) else (
    echo ❌ Build failed!
    echo.
    echo Common issues:
    echo - Make sure you have g++ installed
    echo - Install compression libraries: zlib, lzma, zstd
    echo - Check that melvin_optimized_v2.h and melvin_optimized_v2.cpp exist
)

pause
