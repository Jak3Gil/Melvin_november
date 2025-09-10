@echo off
echo Building Melvin Unified Brain System...
echo.

REM Set up compiler path
set PATH=C:\msys64\mingw64\bin;%PATH%

REM Compile the unified version
echo Compiling melvin_unified.cpp...
g++ -o melvin_unified.exe melvin_unified.cpp -lcurl -std=c++17

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Compilation successful!
    echo Created: melvin_unified.exe
    echo.
    echo Testing the unified system...
    echo.
    echo what is cancer > test_input.txt
    echo quit >> test_input.txt
    echo.
    echo Running Melvin Unified...
    melvin_unified.exe < test_input.txt
    echo.
    echo Cleaning up...
    del test_input.txt
    echo.
    echo 🎉 Melvin Unified Brain System is ready!
) else (
    echo.
    echo ❌ Compilation failed!
    echo Check for errors above.
)

echo.
pause
