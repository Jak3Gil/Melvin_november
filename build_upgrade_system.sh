#!/bin/bash

echo "🔧 Building Melvin Integrated Upgrade System..."
echo "=============================================="

# Compile the integrated upgrade system
g++ -std=c++17 -O2 -o melvin_integrated_upgrade_system melvin_integrated_upgrade_system.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Integrated Upgrade System..."
    echo ""
    
    # Run the system
    ./melvin_integrated_upgrade_system
else
    echo "❌ Build failed!"
    exit 1
fi
