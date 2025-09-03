#!/bin/bash

# 🚀 BUILD MELVIN C++ BRAIN
# Compile high-performance C++ brain components

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 BUILDING MELVIN C++ BRAIN COMPONENTS${NC}"
echo "=================================================="

# Check if we're on macOS or Linux
OS=$(uname -s)
echo -e "🖥️  Operating System: ${GREEN}$OS${NC}"

# Create build directory
BUILD_DIR="build_cpp"
echo -e "📁 Creating build directory: ${YELLOW}$BUILD_DIR${NC}"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Check for required tools
echo -e "\n🔧 Checking build tools..."

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}❌ CMake not found. Please install CMake first.${NC}"
    if [[ "$OS" == "Darwin" ]]; then
        echo "   Install with: brew install cmake"
    else
        echo "   Install with: sudo apt-get install cmake"
    fi
    exit 1
else
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    echo -e "   ✅ CMake found: ${GREEN}$CMAKE_VERSION${NC}"
fi

if ! command -v make &> /dev/null; then
    echo -e "${RED}❌ Make not found. Please install build tools.${NC}"
    exit 1
else
    echo -e "   ✅ Make found: ${GREEN}$(make --version | head -n1)${NC}"
fi

# Check for C++ compiler
if command -v g++ &> /dev/null; then
    CXX_COMPILER="g++"
    CXX_VERSION=$(g++ --version | head -n1)
    echo -e "   ✅ G++ found: ${GREEN}$CXX_VERSION${NC}"
elif command -v clang++ &> /dev/null; then
    CXX_COMPILER="clang++"
    CXX_VERSION=$(clang++ --version | head -n1)
    echo -e "   ✅ Clang++ found: ${GREEN}$CXX_VERSION${NC}"
else
    echo -e "${RED}❌ No C++ compiler found. Please install g++ or clang++.${NC}"
    exit 1
fi

# Check for optional dependencies
echo -e "\n📦 Checking optional dependencies..."

# Python and pybind11 for Python integration
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "   ✅ Python3 found: ${GREEN}$PYTHON_VERSION${NC}"
    
    # Check for pybind11
    if python3 -c "import pybind11" 2>/dev/null; then
        echo -e "   ✅ pybind11 found: ${GREEN}Available${NC}"
        PYTHON_INTEGRATION=ON
    else
        echo -e "   ⚠️  pybind11 not found: ${YELLOW}Python integration disabled${NC}"
        echo "      Install with: pip3 install pybind11"
        PYTHON_INTEGRATION=OFF
    fi
else
    echo -e "   ⚠️  Python3 not found: ${YELLOW}Python integration disabled${NC}"
    PYTHON_INTEGRATION=OFF
fi

# SQLite3
if [[ "$OS" == "Darwin" ]]; then
    # macOS usually has SQLite built-in
    if [ -f /usr/lib/libsqlite3.dylib ] || command -v sqlite3 &> /dev/null; then
        echo -e "   ✅ SQLite3 found: ${GREEN}System library${NC}"
        SQLITE3_AVAILABLE=ON
    else
        echo -e "   ⚠️  SQLite3 not found: ${YELLOW}Database features limited${NC}"
        SQLITE3_AVAILABLE=OFF
    fi
else
    # Linux
    if pkg-config --exists sqlite3 2>/dev/null; then
        SQLITE3_VERSION=$(pkg-config --modversion sqlite3)
        echo -e "   ✅ SQLite3 found: ${GREEN}$SQLITE3_VERSION${NC}"
        SQLITE3_AVAILABLE=ON
    else
        echo -e "   ⚠️  SQLite3 not found: ${YELLOW}Database features limited${NC}"
        echo "      Install with: sudo apt-get install libsqlite3-dev"
        SQLITE3_AVAILABLE=OFF
    fi
fi

# OpenMP for parallel processing
if [[ "$CXX_COMPILER" == "g++" ]] && g++ -fopenmp -dM -E - < /dev/null | grep -q "_OPENMP"; then
    echo -e "   ✅ OpenMP found: ${GREEN}Available with GCC${NC}"
    OPENMP_AVAILABLE=ON
elif [[ "$CXX_COMPILER" == "clang++" ]] && [[ "$OS" == "Darwin" ]]; then
    # macOS Clang usually doesn't have OpenMP by default
    echo -e "   ⚠️  OpenMP not available: ${YELLOW}Parallel processing limited${NC}"
    echo "      Install with: brew install libomp"
    OPENMP_AVAILABLE=OFF
else
    echo -e "   ⚠️  OpenMP status unknown: ${YELLOW}Will be detected by CMake${NC}"
    OPENMP_AVAILABLE=AUTO
fi

# Configure build
echo -e "\n⚙️  Configuring build with CMake..."

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=20
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER
)

# Add Python integration if available
if [[ "$PYTHON_INTEGRATION" == "ON" ]]; then
    CMAKE_ARGS+=(-DBUILD_PYTHON_WRAPPER=ON)
fi

# Run CMake
echo -e "Running: ${YELLOW}cmake ${CMAKE_ARGS[*]} ..${NC}"
cmake "${CMAKE_ARGS[@]}" ..

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ CMake configuration failed${NC}"
    exit 1
fi

echo -e "   ✅ CMake configuration successful"

# Build the project
echo -e "\n🔨 Building C++ components..."

# Detect number of CPU cores for parallel build
if [[ "$OS" == "Darwin" ]]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(nproc)
fi

echo -e "Building with ${GREEN}$CORES${NC} parallel jobs..."
make -j$CORES

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "   ✅ Build successful!"

# Test the build
echo -e "\n🧪 Testing build..."

# Check if Python module was built
if [[ "$PYTHON_INTEGRATION" == "ON" ]] && [ -f "fast_brain_core*.so" ]; then
    echo -e "   ✅ Python module built: ${GREEN}fast_brain_core.so${NC}"
    
    # Test import
    cd ..
    if python3 -c "import sys; sys.path.append('build_cpp'); import fast_brain_core; print('✅ Python module import successful')" 2>/dev/null; then
        echo -e "   ✅ Python module import test: ${GREEN}PASSED${NC}"
        PYTHON_MODULE_WORKING=true
    else
        echo -e "   ⚠️  Python module import test: ${YELLOW}FAILED${NC}"
        PYTHON_MODULE_WORKING=false
    fi
    cd $BUILD_DIR
else
    echo -e "   ⚠️  Python module not built"
    PYTHON_MODULE_WORKING=false
fi

# Performance test
echo -e "\n📊 Performance capabilities detected:"

# Check CPU features
if command -v lscpu &> /dev/null; then
    # Linux
    if lscpu | grep -q "avx2"; then
        echo -e "   ✅ AVX2 SIMD: ${GREEN}Available${NC}"
    else
        echo -e "   ⚠️  AVX2 SIMD: ${YELLOW}Not available${NC}"
    fi
    
    if lscpu | grep -q "sse4_2"; then
        echo -e "   ✅ SSE4.2 SIMD: ${GREEN}Available${NC}"
    else
        echo -e "   ⚠️  SSE4.2 SIMD: ${YELLOW}Not available${NC}"
    fi
elif [[ "$OS" == "Darwin" ]] && command -v sysctl &> /dev/null; then
    # macOS
    if sysctl -a | grep -q "hw.optional.avx2_0: 1"; then
        echo -e "   ✅ AVX2 SIMD: ${GREEN}Available${NC}"
    else
        echo -e "   ⚠️  AVX2 SIMD: ${YELLOW}Not available${NC}"
    fi
    
    if sysctl -a | grep -q "hw.optional.sse4_2: 1"; then
        echo -e "   ✅ SSE4.2 SIMD: ${GREEN}Available${NC}"
    else
        echo -e "   ⚠️  SSE4.2 SIMD: ${YELLOW}Not available${NC}"
    fi
fi

# Summary
echo -e "\n🎯 BUILD SUMMARY"
echo "================================"
echo -e "Build Status: ${GREEN}SUCCESS${NC}"
echo -e "Build Type: ${GREEN}Release (Optimized)${NC}"
echo -e "C++ Standard: ${GREEN}C++20${NC}"
echo -e "Compiler: ${GREEN}$CXX_COMPILER${NC}"
echo -e "CPU Cores Used: ${GREEN}$CORES${NC}"

if [[ "$PYTHON_MODULE_WORKING" == "true" ]]; then
    echo -e "Python Integration: ${GREEN}WORKING${NC}"
else
    echo -e "Python Integration: ${YELLOW}LIMITED${NC}"
fi

echo -e "SQLite3 Support: ${GREEN}$SQLITE3_AVAILABLE${NC}"

# Usage instructions
echo -e "\n🚀 USAGE INSTRUCTIONS"
echo "================================"

if [[ "$PYTHON_MODULE_WORKING" == "true" ]]; then
    echo -e "✅ ${GREEN}C++ brain module is ready!${NC}"
    echo ""
    echo "To use the high-performance C++ brain:"
    echo "  1. Run: python3 melvin_cpp_brain.py"
    echo "  2. Or import in your Python code:"
    echo "     from melvin_cpp_brain import MelvinCppBrain"
    echo ""
    echo "The system will automatically use C++ backend for:"
    echo "  • Ultra-fast node operations"
    echo "  • SIMD-optimized searches"
    echo "  • Parallel Hebbian learning"
    echo "  • Dynamic node optimization"
else
    echo -e "⚠️  ${YELLOW}C++ module built but Python integration limited${NC}"
    echo ""
    echo "To enable full Python integration:"
    echo "  1. Install pybind11: pip3 install pybind11"
    echo "  2. Rebuild: ./build_cpp_brain.sh"
fi

echo ""
echo -e "${GREEN}🎉 Build complete!${NC}"

# Return to original directory
cd ..
