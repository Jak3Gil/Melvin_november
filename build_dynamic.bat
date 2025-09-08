@echo off
echo Building Dynamic Melvin Brain System...
echo.

REM Set up compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

REM Compile the dynamic brain system
echo Compiling melvin_dynamic_brain.cpp...
g++ -o melvin_dynamic.exe melvin_dynamic_brain.cpp -lcurl -std=c++17

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Compilation successful!
    echo Created: melvin_dynamic.exe
    echo.
    echo Testing the dynamic brain system...
    echo.
    echo what do you do if you have cancer > test_input.txt
    echo hello how are you >> test_input.txt
    echo what is artificial intelligence >> test_input.txt
    echo status >> test_input.txt
    echo quit >> test_input.txt
    echo.
    echo Running Dynamic Melvin...
    melvin_dynamic.exe < test_input.txt
    echo.
    echo Cleaning up...
    del test_input.txt
    echo.
    echo 🎉 Dynamic Melvin Brain System is ready!
    echo.
    echo Key Features:
    echo - Pressure-based instincts (no rigid if/else rules)
    echo - Continuous force computation (0.0-1.0)
    echo - Dynamic response generation based on context
    echo - Emotional memory and contextual metadata
    echo - Adaptive style and content based on instinct balance
) else (
    echo.
    echo ❌ Compilation failed!
    echo Check for errors above.
)

echo.
pause
