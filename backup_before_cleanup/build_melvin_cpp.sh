#!/bin/bash

# 🧠 MELVIN OPTIMIZED V2 C++ BUILD SCRIPT
# ======================================

set -e

echo "🧠 Building Melvin Optimized V2 (C++)..."
echo "========================================"

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS"
    
    # Install dependencies using Homebrew
    echo "📦 Installing dependencies..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install required libraries
    brew install cmake pkg-config zlib xz zstd
    
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
            libzlib1g-dev liblzma-dev libzstd-dev
    else
        echo "⚠️  Please install the following packages manually:"
        echo "   - build-essential"
        echo "   - cmake"
        echo "   - pkg-config"
        echo "   - libzlib1g-dev"
        echo "   - liblzma-dev"
        echo "   - libzstd-dev"
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
if [ -f "melvin_optimized_v2_cpp" ]; then
    echo "✅ Build successful!"
    echo "🚀 Executable: ./build/melvin_optimized_v2_cpp"
    
    # Test the executable
    echo "🧪 Testing executable..."
    ./melvin_optimized_v2_cpp
    
    echo ""
    echo "🎉 Melvin Optimized V2 (C++) is ready!"
    echo "📊 Performance improvements expected:"
    echo "   - 10-100x faster processing"
    echo "   - Lower memory overhead"
    echo "   - Better binary handling"
    echo "   - True 4TB optimization"
    
else
    echo "❌ Build failed!"
    exit 1
fi
