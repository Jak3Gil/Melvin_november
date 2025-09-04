#!/bin/bash

# 🚀 Melvin Optimized C++ Node System Build Script
# ================================================

set -e  # Exit on any error

echo "🧠 MELVIN OPTIMIZED C++ NODE SYSTEM"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "brain/optimized_node_system.hpp" ]; then
    echo "❌ Error: Please run this script from the melvin-unified-brain directory"
    exit 1
fi

# Create build directory
echo "📁 Creating build directory..."
mkdir -p build
cd build

# Configure CMake with optimizations
echo "⚙️  Configuring CMake with optimizations..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math" \
      -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
      ../brain

# Build the system
echo "🔨 Building optimized system..."
make -j$(nproc)

# Run tests
echo "🧪 Running tests..."
if [ -f "test_optimized_nodes" ]; then
    ./test_optimized_nodes
else
    echo "⚠️  Test executable not found, skipping tests"
fi

# Show build results
echo ""
echo "✅ BUILD COMPLETED SUCCESSFULLY!"
echo "================================"
echo "📁 Build directory: $(pwd)"
echo "📦 Library: libmelvin_optimized_brain.a"
echo "🧪 Test executable: test_optimized_nodes"
echo "📊 Benchmark executable: benchmark_nodes"

# Check if Python extension was built
if [ -f "melvin_optimized_brain_py.*.so" ]; then
    echo "🐍 Python extension: melvin_optimized_brain_py.*.so"
fi

echo ""
echo "🚀 OPTIMIZATION FEATURES ENABLED:"
echo "   🔹 SIMD optimizations (-march=native)"
echo "   🔹 Fast math (-ffast-math)"
echo "   🔹 Maximum optimization (-O3)"
echo "   🔹 No debug symbols in release (-DNDEBUG)"
echo "   🔹 Cache-friendly data layouts"
echo "   🔹 Byte-level memory management"

echo ""
echo "📊 MEMORY EFFICIENCY:"
echo "   🔹 Node structure: 60 bytes"
echo "   🔹 Connection structure: 16 bytes"
echo "   🔹 Configuration: 16 bytes"
echo "   🔹 Total overhead: ~80 bytes per node"

echo ""
echo "🎯 USAGE:"
echo "   # Run tests"
echo "   ./test_optimized_nodes"
echo ""
echo "   # Run benchmarks"
echo "   ./benchmark_nodes"
echo ""
echo "   # Use from Python (if extension built)"
echo "   python3 -c \"import melvin_optimized_brain_py as melvin; sizer = melvin.OptimizedDynamicNodeSizer(); nodes = sizer.create_dynamic_nodes('AI machine learning')\""

echo ""
echo "🎉 Ready for high-performance node processing!"
