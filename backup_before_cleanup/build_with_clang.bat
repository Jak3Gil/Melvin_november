@echo off
echo 🛠️  MELVIN COMPILATION WITH CLANG
echo =================================
echo Setting up environment and compiling Melvin...

REM Add LLVM to PATH
set PATH=%PATH%;"C:\Program Files\LLVM\bin"

REM Try to find Visual Studio environment
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio Build Tools, setting up environment...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    echo Found Visual Studio Build Tools, setting up environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo Visual Studio environment not found, trying direct compilation...
)

echo.
echo Compiling simplified interactive Melvin...
clang-cl /O2 /std:c++17 /EHsc -o melvin_simple_interactive.exe melvin_simple_interactive.cpp melvin_optimized_v2.cpp

if %ERRORLEVEL% EQU 0 (
    echo ✅ Compilation successful!
    echo.
    echo Testing the executable...
    melvin_simple_interactive.exe
) else (
    echo ❌ Compilation failed!
    echo.
    echo Trying alternative compilation with g++-style flags...
    clang++ -O2 -std=c++17 -o melvin_simple_interactive.exe melvin_simple_interactive.cpp melvin_optimized_v2.cpp
    
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Alternative compilation successful!
        echo.
        echo Testing the executable...
        melvin_simple_interactive.exe
    ) else (
        echo ❌ All compilation attempts failed!
        echo.
        echo The issue is likely missing Windows SDK headers.
        echo You may need to install Visual Studio Community or use a different approach.
    )
)

echo.
pause
