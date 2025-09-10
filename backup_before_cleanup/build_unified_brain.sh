#!/bin/bash

# 🧠 MELVIN UNIFIED BRAIN BUILD SCRIPT
# ====================================

set -e

echo "🧠 Building Melvin Unified Brain System..."
echo "=========================================="

# Check if we're on Windows
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "🪟 Detected Windows"
    
    # Check for MinGW or Visual Studio
    if command -v g++ &> /dev/null; then
        echo "📦 Using MinGW compiler"
        COMPILER="g++"
    elif command -v cl &> /dev/null; then
        echo "📦 Using Visual Studio compiler"
        COMPILER="cl"
    else
        echo "❌ No suitable compiler found. Please install MinGW or Visual Studio."
        exit 1
    fi
    
    # Install dependencies using vcpkg or manual installation
    echo "📦 Installing dependencies..."
    echo "Please ensure you have:"
    echo "- libcurl (for web search)"
    echo "- nlohmann/json (for JSON parsing)"
    echo "- zlib (for compression)"
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS"
    
    # Install dependencies using Homebrew
    echo "📦 Installing dependencies..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install required libraries
    brew install cmake pkg-config zlib curl nlohmann-json
    
    # Set environment variables for macOS
    export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
    export LDFLAGS="-L/opt/homebrew/lib -L/usr/local/lib"
    export CPPFLAGS="-I/opt/homebrew/include -I/usr/local/include"
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Detected Linux"
    
    # Install dependencies using apt (Ubuntu/Debian)
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing dependencies..."
        sudo apt-get update
        sudo apt-get install -y build-essential cmake pkg-config \
            libzlib1g-dev libcurl4-openssl-dev nlohmann-json3-dev
    else
        echo "⚠️  Please install the following packages manually:"
        echo "   - build-essential"
        echo "   - cmake"
        echo "   - pkg-config"
        echo "   - libzlib1g-dev"
        echo "   - libcurl4-openssl-dev"
        echo "   - nlohmann-json3-dev"
    fi
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

# Create build directory
echo "📁 Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "⚙️  Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "🔨 Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if [ -f "melvin_unified_brain" ]; then
    echo "✅ Build successful!"
    echo "🚀 Executable: ./build/melvin_unified_brain"
    
    # Test the executable
    echo "🧪 Testing executable..."
    echo "Setting up environment..."
    
    # Check for Bing API key
    if [ -z "$BING_API_KEY" ]; then
        echo "⚠️  BING_API_KEY environment variable not set."
        echo "   Web search functionality will be limited."
        echo "   Set it with: export BING_API_KEY='your_api_key_here'"
    else
        echo "✅ BING_API_KEY found - web search enabled"
    fi
    
    echo ""
    echo "🎉 Melvin Unified Brain System is ready!"
    echo "📊 Features:"
    echo "   - Binary node memory with 28-byte headers"
    echo "   - Hebbian learning connections"
    echo "   - Instinct-driven reasoning"
    echo "   - Web search integration"
    echo "   - Transparent reasoning paths"
    echo "   - Dynamic learning and growth"
    echo ""
    echo "🚀 Run with: ./build/melvin_unified_brain"
    echo "📖 Commands: 'status', 'help', 'memory', 'instincts', 'learn'"
    
else
    echo "❌ Build failed!"
    exit 1
fi
