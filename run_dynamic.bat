@echo off
echo Starting Dynamic Melvin Brain System...
echo.

if not exist melvin_dynamic.exe (
    echo melvin_dynamic.exe not found. Building first...
    call build_dynamic.bat
    if %ERRORLEVEL% NEQ 0 (
        echo Build failed. Exiting.
        pause
        exit /b 1
    )
)

echo Starting dynamic interactive session...
echo.
echo This version features:
echo - Pressure-based instincts (no rigid rules)
echo - Continuous force computation
echo - Dynamic response generation
echo - Emotional memory and context
echo - Adaptive style based on instinct balance
echo.
melvin_dynamic.exe

echo.
echo Dynamic session ended.
pause
