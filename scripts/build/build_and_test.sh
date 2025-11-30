#!/bin/bash
# Build and test Melvin on Linux
# This script should be run inside the Linux VM

set -e

echo "=========================================="
echo "Melvin Build and Test on Linux"
echo "=========================================="
echo ""

# Install dependencies
echo "📦 Installing build dependencies..."
sudo apt update
sudo apt install -y build-essential gcc make 2>&1 | grep -E "(Setting up|is already)" || true
echo ""

# Compile the stub test
echo "🔨 Compiling test_exec_stub..."
gcc -o test_exec_stub test_exec_stub.c -lm -std=c11 -Wall -Wextra || {
    echo "❌ Compilation failed"
    exit 1
}
echo "✅ Compilation successful"
echo ""

# Run the stub test
echo "🧪 Running EXEC stub test..."
echo ""
./test_exec_stub

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="

