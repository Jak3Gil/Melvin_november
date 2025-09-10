@echo off
echo Building Simple Melvin Brain System...
echo.

REM Set up compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

REM Compile the simple version
echo Compiling melvin_simple.cpp...
g++ -o melvin_simple.exe melvin_simple.cpp -std=c++17

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Compilation successful!
    echo Created: melvin_simple.exe
    echo.
    echo Testing the simple brain system...
    echo.
    echo what is cancer > test_input.txt
    echo hello how are you >> test_input.txt
    echo what is a dog >> test_input.txt
    echo quit >> test_input.txt
    echo.
    echo Running Simple Melvin...
    melvin_simple.exe < test_input.txt
    echo.
    echo Cleaning up...
    del test_input.txt
    echo.
    echo 🎉 Simple Melvin Brain System is ready!
    echo.
    echo You can now run: melvin_simple.exe
    echo Or use: run_simple.bat
) else (
    echo.
    echo ❌ Compilation failed!
    echo Check for errors above.
)

echo.
pause
