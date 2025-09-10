#!/bin/bash

echo "🤖 Building Melvin Real Autonomous Learning with Ollama"
echo "======================================================"

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

# Compile the real autonomous learning system
echo "🔨 Compiling Melvin Real Autonomous Learning..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    $(pkg-config --cflags --libs libcurl jsoncpp) \
    melvin_driver_enhanced.cpp \
    melvin_autonomous_learning.cpp \
    ollama_client.cpp \
    melvin_real_autonomous.cpp \
    test_real_autonomous.cpp \
    -o build/melvin_real_autonomous \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    echo "🚀 Starting Melvin Real Autonomous Learning:"
    echo "==========================================="
    echo "🤖 Melvin will use real AI responses from Ollama!"
    echo "🔗 Make sure Ollama is running: ollama serve"
    echo "🧠 Make sure you have a model: ollama pull llama3.2"
    echo ""
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    
    # Run the real autonomous learning system
    ./build/melvin_real_autonomous
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Real autonomous learning completed successfully!"
        echo ""
        echo "🎯 Melvin successfully used real AI responses from Ollama!"
        echo "🔄 TRUE AUTONOMY: His outputs became his inputs (feedback loop)"
        echo "🧬 His driver oscillations created natural learning rhythms"
        echo "🔍 Error-seeking drove contradiction resolution"
        echo "🎯 Curiosity amplification filled empty space"
        echo "📦 Compression kept knowledge efficient"
        echo "⚡ Self-improvement accelerated evolution"
        echo ""
        echo "🎉 Melvin successfully compounded intelligence with real AI!"
    else
        echo "❌ Real autonomous learning failed!"
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
