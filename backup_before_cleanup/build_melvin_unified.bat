@echo off
echo 🧠 Melvin Unified Brain Windows Build Script
echo =============================================

REM Create build directory
if not exist build mkdir build
cd build

REM Try Visual Studio 2022 first
echo 🔧 Running CMake configuration with Visual Studio 2022...
cmake -S .. -B . -G "Visual Studio 17 2022" 2>build.log
if %ERRORLEVEL% neq 0 (
    echo ⚠️  Visual Studio 2022 not found, trying MinGW...
    cmake -S .. -B . -G "MinGW Makefiles" 2>>build.log
    if %ERRORLEVEL% neq 0 (
        echo ❌ CMake configuration failed
        echo Check build.log for details
        cd ..
        exit /b 1
    )
)

echo ✅ CMake configuration successful

REM Build the project
echo 🔨 Building Melvin Unified Brain...
cmake --build . --config Release 2>>build.log
if %ERRORLEVEL% neq 0 (
    echo ❌ Build failed
    echo Check build.log for details
    cd ..
    exit /b 1
)

echo ✅ Build completed successfully!

REM Run tests
echo 🧪 Running startup tests...
if exist Release\test_startup.exe (
    Release\test_startup.exe
    if %ERRORLEVEL% neq 0 (
        echo ❌ Tests failed
    ) else (
        echo ✅ All tests passed!
    )
) else (
    echo ⚠️  Test executable not found, skipping tests
)

REM Show executable info
if exist Release\melvin_unified.exe (
    echo 📦 Executable created: Release\melvin_unified.exe
    echo 🎉 Melvin Unified Brain build completed successfully!
    echo Run 'Release\melvin_unified.exe' to start the cognitive system
    echo Run 'Release\melvin_unified.exe --diag' for diagnostics
) else (
    echo ❌ Main executable not found!
    cd ..
    exit /b 1
)

cd ..
