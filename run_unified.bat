@echo off
echo Starting Melvin Unified Brain System...
echo.

if not exist melvin_unified.exe (
    echo melvin_unified.exe not found. Building first...
    call build_unified.bat
    if %ERRORLEVEL% NEQ 0 (
        echo Build failed. Exiting.
        pause
        exit /b 1
    )
)

echo Starting interactive session...
echo.
melvin_unified.exe

echo.
echo Session ended.
pause
