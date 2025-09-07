@echo off
echo 🛠️  PERMANENT PATH SETUP FOR C++ COMPILER
echo =========================================
echo This script will add MinGW-w64 to your system PATH permanently
echo Run this as Administrator for best results
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Not running as Administrator
    echo This may limit what can be changed.
    echo.
)

REM Check if MinGW exists
if exist "C:\msys64\mingw64\bin\g++.exe" (
    echo ✅ MinGW-w64 found at C:\msys64\mingw64\bin
) else (
    echo ❌ MinGW-w64 not found at expected location
    echo Please install MinGW-w64 first using install_mingw.bat
    pause
    exit /b 1
)

REM Get current PATH
for /f "tokens=2*" %%A in ('reg query "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "CURRENT_PATH=%%B"

REM Check if MinGW is already in PATH
echo %CURRENT_PATH% | findstr /i "C:\msys64\mingw64\bin" >nul
if %errorlevel% equ 0 (
    echo ✅ MinGW-w64 is already in system PATH
    echo.
    echo Testing compiler...
    set PATH=C:\msys64\mingw64\bin;%PATH%
    g++ --version
    echo.
    echo You can now use g++ from any command prompt!
    pause
    exit /b 0
)

echo Adding MinGW-w64 to system PATH...
echo Current PATH: %CURRENT_PATH%
echo.

REM Add MinGW to PATH
set "NEW_PATH=%CURRENT_PATH%;C:\msys64\mingw64\bin"

REM Update system PATH
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH /t REG_EXPAND_SZ /d "%NEW_PATH%" /f

if %errorlevel% equ 0 (
    echo ✅ Successfully added MinGW-w64 to system PATH!
    echo.
    echo IMPORTANT: You need to restart your command prompt or computer
    echo for the changes to take effect.
    echo.
    echo After restarting, you can use g++ from anywhere.
) else (
    echo ❌ Failed to update system PATH
    echo You may need to run this script as Administrator
    echo.
    echo Manual alternative:
    echo 1. Open System Properties ^> Advanced ^> Environment Variables
    echo 2. Edit the PATH variable
    echo 3. Add: C:\msys64\mingw64\bin
)

echo.
echo Testing compiler in current session...
set PATH=C:\msys64\mingw64\bin;%PATH%
g++ --version

pause
