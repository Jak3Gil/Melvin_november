@echo off
REM 🔍 BING API KEY SETUP FOR MELVIN
REM =================================

echo 🔍 Bing API Key Setup for Melvin
echo =================================
echo.

echo Step 1: Get your Bing API key from Azure
echo -----------------------------------------
echo 1. Go to: https://portal.azure.com/
echo 2. Create a Bing Search v7 resource (Free tier: 1000 queries/month)
echo 3. Copy your API key from "Keys and Endpoint"
echo.

echo Step 2: Enter your API key below
echo --------------------------------
set /p API_KEY="Enter your Bing API key: "

if "%API_KEY%"=="" (
    echo ❌ No API key entered. Exiting.
    pause
    exit /b 1
)

echo.
echo Setting environment variable...
set BING_API_KEY=%API_KEY%

echo ✅ API key set for this session!
echo.
echo Testing Melvin with web search...
echo.

melvin_bing.exe

echo.
echo To make this permanent, add BING_API_KEY to your system environment variables.
echo.
pause
