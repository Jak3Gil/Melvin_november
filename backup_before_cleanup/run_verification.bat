@echo off
echo 🧠 Compiling Melvin Blended Reasoning Verification...
echo ==================================================

REM Try to compile with available compiler
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" (
    echo Using Visual Studio 2022 compiler...
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" /std:c++17 verify_blended_reasoning.cpp melvin_optimized_v2.cpp /Fe:verify_blended_reasoning.exe
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" (
    echo Using Visual Studio 2019 compiler...
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" /std:c++17 verify_blended_reasoning.cpp melvin_optimized_v2.cpp /Fe:verify_blended_reasoning.exe
) else (
    echo No Visual Studio compiler found. Trying alternative approach...
    echo Checking if MinGW or other compiler is available...
    where gcc >nul 2>&1
    if %errorlevel% == 0 (
        echo Using GCC compiler...
        gcc -std=c++17 -o verify_blended_reasoning.exe verify_blended_reasoning.cpp melvin_optimized_v2.cpp -lstdc++
    ) else (
        echo No suitable compiler found. Please install Visual Studio or MinGW.
        pause
        exit /b 1
    )
)

if %errorlevel% == 0 (
    echo ✅ Compilation successful!
    echo.
    echo 🚀 Running Melvin Blended Reasoning Verification...
    echo ==================================================
    verify_blended_reasoning.exe
) else (
    echo ❌ Compilation failed!
    echo Please ensure you have a C++ compiler installed.
)

pause
