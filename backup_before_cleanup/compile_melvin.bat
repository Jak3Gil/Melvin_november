@echo off
echo 🛠️  MELVIN COMPILER SETUP
echo =========================
echo Setting up C++ compiler and compiling Melvin
echo.

REM Set up MinGW path for this session
set PATH=C:\msys64\mingw64\bin;%PATH%

echo Checking for compiler...
where g++ >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ g++ not found. Checking MSYS2 installation...
    
    if exist "C:\msys64\mingw64\bin\g++.exe" (
        echo ✅ Found g++ at C:\msys64\mingw64\bin\g++.exe
        echo Adding to PATH for this session...
    ) else (
        echo ❌ MinGW compiler not installed.
        echo.
        echo To install MinGW-w64:
        echo 1. Open MSYS2 terminal: C:\msys64\msys2.exe
        echo 2. Run: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gdb
        echo 3. Run this script again
        pause
        exit /b 1
    )
)

echo ✅ Compiler ready!
g++ --version
echo.

echo Compiling Melvin Simple Working version...
g++ -std=c++17 -O2 -o melvin_simple_working.exe melvin_simple_working.cpp

if %errorlevel% equ 0 (
    echo ✅ Compilation successful!
    echo.
    echo Running Melvin...
    echo =================
    melvin_simple_working.exe
) else (
    echo ❌ Compilation failed!
    echo Check the error messages above.
)

echo.
pause
