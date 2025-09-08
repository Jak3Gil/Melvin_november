@echo off
echo 🛠️  MINGW-W64 COMPILER INSTALLER
echo ================================
echo Installing C++ compiler for Windows
echo.

REM Check if MinGW is already installed
if exist "C:\msys64\mingw64\bin\g++.exe" (
    echo ✅ MinGW-w64 is already installed!
    echo Location: C:\msys64\mingw64\bin\g++.exe
    echo.
    echo Adding to PATH...
    call add_compiler_to_path.bat
    exit /b 0
)

echo Checking for MSYS2 installation...
if exist "C:\msys64\msys2.exe" (
    echo ✅ MSYS2 found at C:\msys64
    echo.
    echo Installing MinGW-w64 compiler...
    echo This may take a few minutes...
    echo.
    
    REM Install MinGW through MSYS2
    C:\msys64\usr\bin\bash.exe -c "pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gdb --noconfirm"
    
    if %errorlevel% equ 0 (
        echo ✅ MinGW-w64 installed successfully!
        echo.
        echo Adding to PATH...
        call add_compiler_to_path.bat
    ) else (
        echo ❌ Installation failed. Please try manual installation.
        echo.
        echo Manual steps:
        echo 1. Open MSYS2 terminal: C:\msys64\msys2.exe
        echo 2. Run: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gdb
        echo 3. Run: add_compiler_to_path.bat
    )
) else (
    echo ❌ MSYS2 not found. Installing MSYS2 first...
    echo.
    echo Please download and install MSYS2 from:
    echo https://www.msys2.org/
    echo.
    echo After installation, run this script again.
)

pause
