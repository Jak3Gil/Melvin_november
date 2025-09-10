#!/bin/bash

echo "🔧 MELVIN REAL API UPGRADE BUILD SCRIPT"
echo "========================================"

# Check for required dependencies
echo "Checking dependencies..."

# Check for libcurl
if ! pkg-config --exists libcurl; then
    echo "❌ libcurl not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y libcurl4-openssl-dev
    elif command -v yum &> /dev/null; then
        sudo yum install -y libcurl-devel
    elif command -v brew &> /dev/null; then
        brew install curl
    else
        echo "❌ Please install libcurl manually for your system"
        exit 1
    fi
fi

# Check for nlohmann/json
if ! pkg-config --exists nlohmann_json; then
    echo "❌ nlohmann/json not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y nlohmann-json3-dev
    elif command -v yum &> /dev/null; then
        sudo yum install -y nlohmann-json-devel
    elif command -v brew &> /dev/null; then
        brew install nlohmann-json
    else
        echo "❌ Please install nlohmann-json manually for your system"
        exit 1
    fi
fi

echo "✅ Dependencies checked"

# Compile with real API support
echo "🔨 Compiling Melvin with real API support..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I/usr/include \
    -I/usr/local/include \
    melvin_interactive.cpp \
    -lcurl -lnlohmann_json \
    -o melvin_interactive_api.exe

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 SETUP INSTRUCTIONS:"
    echo "1. Get a Bing Search API key from: https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/"
    echo "2. Set environment variable: export BING_API_KEY='your_api_key_here'"
    echo "3. Run: ./melvin_interactive_api.exe"
    echo ""
    echo "🧪 TEST COMMAND:"
    echo "echo 'What are carbon nanotubes?' | ./melvin_interactive_api.exe"
else
    echo "❌ Compilation failed!"
    exit 1
fi
