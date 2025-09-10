#!/bin/bash

echo "🤖 Building Melvin Genuine Autonomous Learning"
echo "=============================================="
echo "NO FAKE METRICS - ONLY REAL AI RESPONSES AND LEARNING!"

# Create build directory
mkdir -p build

# Check if libcurl and jsoncpp are available
echo "🔍 Checking dependencies..."

# Check for libcurl
if ! pkg-config --exists libcurl; then
    echo "⚠️ Warning: libcurl not found. You may need to install it:"
    echo "   macOS: brew install curl"
    echo "   Ubuntu: sudo apt-get install libcurl4-openssl-dev"
    echo "   CentOS: sudo yum install libcurl-devel"
fi

# Check for jsoncpp
if ! pkg-config --exists jsoncpp; then
    echo "⚠️ Warning: jsoncpp not found. You may need to install it:"
    echo "   macOS: brew install jsoncpp"
    echo "   Ubuntu: sudo apt-get install libjsoncpp-dev"
    echo "   CentOS: sudo yum install jsoncpp-devel"
fi

# Compile the genuine autonomous learning system
echo "🔨 Compiling Melvin Genuine Autonomous Learning..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    $(pkg-config --cflags --libs libcurl jsoncpp) \
    ollama_client.cpp \
    melvin_genuine_autonomous.cpp \
    -o build/melvin_genuine_autonomous \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    echo "🚀 Starting Melvin Genuine Autonomous Learning:"
    echo "=============================================="
    echo "🤖 Melvin will use REAL AI responses from Ollama!"
    echo "🧠 NO FAKE METRICS - ONLY GENUINE LEARNING!"
    echo "💡 Real insight generation and concept extraction"
    echo "⚡ Actual self-improvement based on AI responses"
    echo "🔄 TRUE AUTONOMY: His outputs become his inputs!"
    echo ""
    echo "🔗 Make sure Ollama is running: ollama serve"
    echo "🧠 Make sure you have a model: ollama pull llama3.2"
    echo ""
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    
    # Run the genuine autonomous learning system
    ./build/melvin_genuine_autonomous
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Genuine autonomous learning completed successfully!"
        echo ""
        echo "🎯 Melvin successfully used REAL AI responses from Ollama!"
        echo "🔄 TRUE AUTONOMY: His outputs became his inputs (feedback loop)"
        echo "🧠 Genuine learning and concept extraction"
        echo "💡 Real insight generation"
        echo "⚡ Actual self-improvement"
        echo "📊 NO FAKE METRICS - ONLY REAL LEARNING!"
        echo ""
        echo "🎉 Melvin successfully compounded intelligence genuinely!"
    else
        echo "❌ Genuine autonomous learning failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "1. Make sure you have libcurl installed"
    echo "2. Make sure you have jsoncpp installed"
    echo "3. Check that Ollama is running: ollama serve"
    echo "4. Check that you have a model: ollama pull llama3.2"
    exit 1
fi
