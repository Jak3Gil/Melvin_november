@echo off
echo 🛠️  MELVIN C++ COMPILER SETUP FOR WINDOWS
echo ========================================
echo This script will help you set up a C++ compiler for Melvin
echo.

echo Checking current compiler status...
where g++ >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ MinGW-w64 g++ is already installed and available
    g++ --version
    echo.
    echo You can now run: run_improved_demo.bat
    pause
    exit /b 0
)

echo ❌ No C++ compiler found. Setting up MinGW-w64...
echo.

echo Option 1: Download and install MinGW-w64 manually
echo ================================================
echo 1. Go to: https://www.mingw-w64.org/downloads/
echo 2. Download "MingW-W64-builds" 
echo 3. Run the installer
echo 4. Add C:\mingw64\bin to your PATH environment variable
echo.

echo Option 2: Use Chocolatey (if installed)
echo ======================================
where choco >nul 2>&1
if %errorlevel% equ 0 (
    echo Found Chocolatey! Installing MinGW-w64...
    choco install mingw -y
    echo.
    echo Please restart your command prompt and run this script again.
    pause
    exit /b 0
)

echo Option 3: Use winget (Windows Package Manager)
echo ============================================
where winget >nul 2>&1
if %errorlevel% equ 0 (
    echo Found winget! Installing MinGW-w64...
    winget install -e --id MSYS2.MSYS2
    echo.
    echo After installation, open MSYS2 terminal and run:
    echo   pacman -S mingw-w64-x86_64-gcc
    echo   pacman -S mingw-w64-x86_64-gdb
    echo.
    echo Then add C:\msys64\mingw64\bin to your PATH
    pause
    exit /b 0
)

echo No package manager found. Please install MinGW-w64 manually:
echo.
echo 1. Download from: https://www.mingw-w64.org/downloads/
echo 2. Install to C:\mingw64
echo 3. Add C:\mingw64\bin to your PATH environment variable
echo 4. Restart your command prompt
echo 5. Run this script again to verify installation
echo.
pause
