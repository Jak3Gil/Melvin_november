@echo off
echo Starting Simple Melvin Brain System...
echo.

if not exist melvin_simple.exe (
    echo melvin_simple.exe not found. Building first...
    call build_simple.bat
    if %ERRORLEVEL% NEQ 0 (
        echo Build failed. Exiting.
        pause
        exit /b 1
    )
)

echo Starting simple interactive session...
echo.
melvin_simple.exe

echo.
echo Simple session ended.
pause
