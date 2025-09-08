@echo off
REM 🧠 MELVIN UNIFIED BRAIN BUILD SCRIPT (Windows)
REM ==============================================

echo 🧠 Building Melvin Unified Brain System...
echo ==========================================

REM Check for MinGW
where g++ >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo 📦 Using MinGW compiler
    set COMPILER=g++
    goto :build
)

REM Check for Visual Studio
where cl >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo 📦 Using Visual Studio compiler
    set COMPILER=cl
    goto :build
)

echo ❌ No suitable compiler found. Please install MinGW or Visual Studio.
exit /b 1

:build
echo 📦 Installing dependencies...
echo Please ensure you have:
echo - libcurl (for web search)
echo - nlohmann/json (for JSON parsing)
echo - zlib (for compression)

REM Create build directory
echo 📁 Creating build directory...
if not exist build mkdir build
cd build

REM Configure with CMake
echo ⚙️  Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 (
    echo ❌ CMake configuration failed!
    exit /b 1
)

REM Build
echo 🔨 Building...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed!
    exit /b 1
)

REM Check if build was successful
if exist "Release\melvin_unified_brain.exe" (
    echo ✅ Build successful!
    echo 🚀 Executable: .\build\Release\melvin_unified_brain.exe
    
    REM Test the executable
    echo 🧪 Testing executable...
    echo Setting up environment...
    
    REM Check for Bing API key
    if "%BING_API_KEY%"=="" (
        echo ⚠️  BING_API_KEY environment variable not set.
        echo    Web search functionality will be limited.
        echo    Set it with: set BING_API_KEY=your_api_key_here
    ) else (
        echo ✅ BING_API_KEY found - web search enabled
    )
    
    echo.
    echo 🎉 Melvin Unified Brain System is ready!
    echo 📊 Features:
    echo    - Binary node memory with 28-byte headers
    echo    - Hebbian learning connections
    echo    - Instinct-driven reasoning
    echo    - Web search integration
    echo    - Transparent reasoning paths
    echo    - Dynamic learning and growth
    echo.
    echo 🚀 Run with: .\build\Release\melvin_unified_brain.exe
    echo 📖 Commands: 'status', 'help', 'memory', 'instincts', 'learn'
    
) else (
    echo ❌ Build failed!
    exit /b 1
)

pause
