@echo off
echo 🧠 MELVIN IMPROVED WEB SEARCH DEMONSTRATION
echo ===========================================
echo Testing improved web search with clearer responses
echo No Python dependencies - Pure C++ implementation
echo.

REM Try to find available C++ compiler
set COMPILER_FOUND=0

REM Check for Visual Studio compiler
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %errorlevel% equ 0 (
    echo Found Visual Studio compiler
    set COMPILER=cl
    set COMPILER_FOUND=1
    goto :compile
)

REM Check for MinGW-w64
where g++ >nul 2>&1
if %errorlevel% equ 0 (
    echo Found MinGW-w64 g++
    set COMPILER=g++
    set COMPILER_FOUND=1
    goto :compile
)

REM Check for TDM-GCC
where gcc >nul 2>&1
if %errorlevel% equ 0 (
    echo Found GCC compiler
    set COMPILER=gcc
    set COMPILER_FOUND=1
    goto :compile
)

REM If no compiler found, provide instructions
echo ❌ No C++ compiler found!
echo.
echo Please install one of the following:
echo 1. Visual Studio Build Tools 2022 with C++ workload
echo 2. MinGW-w64 from https://www.mingw-w64.org/
echo 3. TDM-GCC from https://jmeubank.github.io/tdm-gcc/
echo.
echo After installation, restart this script.
pause
exit /b 1

:compile
echo Compiling melvin_improved_demo.cpp using %COMPILER%...

if "%COMPILER%"=="cl" (
    cl /EHsc /std:c++17 /O2 melvin_improved_demo.cpp melvin_optimized_v2.cpp /Fe:melvin_improved_demo.exe
) else (
    %COMPILER% -std=c++17 -O2 -o melvin_improved_demo.exe melvin_improved_demo.cpp melvin_optimized_v2.cpp
)

if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Compilation successful. Running demo...
melvin_improved_demo.exe
pause
