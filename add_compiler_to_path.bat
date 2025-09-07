@echo off
echo 🛠️  ADDING C++ COMPILER TO SYSTEM PATH
echo =====================================
echo This will add MinGW-w64 to your system PATH permanently
echo.

REM Check if MinGW is already in PATH
where g++ >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ g++ compiler is already available in PATH
    g++ --version
    echo.
    echo You can now use g++ from any command prompt!
    pause
    exit /b 0
)

REM Add MinGW to PATH for current session
set PATH=C:\msys64\mingw64\bin;%PATH%

REM Test if compiler is available
where g++ >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ MinGW compiler not found at C:\msys64\mingw64\bin
    echo.
    echo Please install MinGW-w64 first:
    echo 1. Open MSYS2 terminal (C:\msys64\msys2.exe)
    echo 2. Run: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gdb
    echo 3. Run this script again
    pause
    exit /b 1
)

echo ✅ Found MinGW compiler!
g++ --version
echo.

echo Adding MinGW to system PATH permanently...
echo This requires administrator privileges.

REM Add to system PATH permanently
setx PATH "%PATH%" /M

if %errorlevel% equ 0 (
    echo ✅ Successfully added MinGW to system PATH!
    echo.
    echo Please restart your command prompt for changes to take effect.
    echo After restarting, you can use g++ from anywhere.
) else (
    echo ❌ Failed to add to system PATH. You may need to run as administrator.
    echo.
    echo Alternative: Add C:\msys64\mingw64\bin to your PATH manually:
    echo 1. Open System Properties ^> Advanced ^> Environment Variables
    echo 2. Edit PATH variable
    echo 3. Add: C:\msys64\mingw64\bin
)

echo.
echo Testing compiler in current session...
g++ --version
echo.
pause
