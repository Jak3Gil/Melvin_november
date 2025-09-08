@echo off
REM 🧪 TEST BING API INTEGRATION
REM ============================

echo 🧪 Testing Bing API Integration
echo ============================
echo.

REM Check if API key is set
if "%BING_API_KEY%"=="" (
    echo ❌ BING_API_KEY not set!
    echo.
    echo Please run: setup_bing_key.bat
    echo Or set manually: set BING_API_KEY=your_key_here
    echo.
    pause
    exit /b 1
)

echo ✅ BING_API_KEY is set: %BING_API_KEY:~0,8%...
echo.

echo Starting Melvin with web search enabled...
echo Ask him: "What is quantum computing?" or "Latest AI news"
echo.
echo Type 'quit' to exit when done testing.
echo.

melvin_bing.exe

echo.
echo Test complete!
pause
