@echo off
REM 🧠 MELVIN BING API SETUP
REM ========================

echo 🧠 Melvin Bing API Setup
echo ========================
echo.
echo To enable real web search, you need a Bing Search API key.
echo.
echo 1. Go to: https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/
echo 2. Create a free Azure account (if you don't have one)
echo 3. Create a Bing Search resource
echo 4. Get your API key from the Azure portal
echo.
echo Then set your API key:
echo.
echo   set BING_API_KEY=your_api_key_here
echo.
echo Or add it permanently to your environment variables.
echo.
echo After setting the API key, run: melvin_bing.exe
echo.
echo Without the API key, Melvin will use his knowledge base only.
echo.
pause
