#!/bin/bash

# Melvin Unified Brain - Clean Build Script
# This script builds the core Melvin system components

echo "🧠 Building Melvin Unified Brain System..."

# Create build directory
mkdir -p build

# Build main components
echo "📦 Building core components..."

# Build autonomous learning system
g++ -std=c++17 -O3 -march=native -ffast-math -DNDEBUG \
    -o build/melvin_autonomous_learning \
    melvin_autonomous_learning.cpp \
    melvin_driver_enhanced.cpp \
    -lz -llzma -lzstd

# Build unified brain system
g++ -std=c++17 -O3 -march=native -ffast-math -DNDEBUG \
    -o build/melvin_unified_brain \
    melvin_unified_brain.cpp \
    melvin_unified_system.cpp \
    -lz -llzma -lzstd

# Build optimized core system
g++ -std=c++17 -O3 -march=native -ffast-math -DNDEBUG \
    -o build/melvin_optimized_v2 \
    melvin_optimized_v2.cpp \
    -lz -llzma -lzstd

# Build CLI interface
g++ -std=c++17 -O3 -march=native -ffast-math -DNDEBUG \
    -o build/melvin_cli \
    melvin_cli.cpp \
    melvin_optimized_v2.cpp \
    -lz -llzma -lzstd

# Build brain monitor
g++ -std=c++17 -O3 -march=native -ffast-math -DNDEBUG \
    -o build/melvin_brain_monitor \
    melvin_brain_monitor.cpp \
    melvin_optimized_v2.cpp \
    -lz -llzma -lzstd

# Build data feeder
g++ -std=c++17 -O3 -march=native -ffast-math -DNDEBUG \
    -o build/feed_melvin_data \
    feed_melvin_data.cpp \
    melvin_optimized_v2.cpp \
    -lz -llzma -lzstd

echo "✅ Build complete! Executables created in build/ directory"
echo ""
echo "🚀 Usage:"
echo "  ./build/melvin_autonomous_learning  # Run autonomous learning system"
echo "  ./build/melvin_unified_brain        # Run unified brain system"
echo "  ./build/melvin_optimized_v2         # Run optimized core system"
echo "  ./build/melvin_cli                 # Run CLI interface"
echo "  ./build/melvin_brain_monitor        # Monitor brain activity"
echo "  ./build/feed_melvin_data            # Feed data to Melvin"
