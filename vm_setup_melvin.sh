#!/bin/bash
# Run this INSIDE the Linux VM to set up Melvin

set -e

echo "=========================================="
echo "Melvin Setup in Linux VM"
echo "=========================================="
echo ""

# Check we're in Linux
if ! uname -a | grep -qi linux; then
    echo "⚠️  Warning: Doesn't look like Linux. Are you in the VM?"
fi

echo "Linux Kernel Info:"
uname -a
echo ""

# Check IP
echo "VM IP Address:"
ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print "  " $2}' || echo "  (checking...)"
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt update -qq
echo ""

# Install build tools
echo "🔧 Installing build tools..."
sudo apt install -y build-essential gcc make git 2>&1 | tail -5
echo "✅ Build tools installed"
echo ""

# Check if Melvin directory exists
if [ -d "$HOME/melvin_november" ]; then
    echo "✅ Melvin directory found: ~/melvin_november"
    cd ~/melvin_november
else
    echo "📁 Creating Melvin directory..."
    mkdir -p ~/melvin_november
    cd ~/melvin_november
    echo ""
    echo "⚠️  Melvin files not found."
    echo "   Transfer from Mac using:"
    echo "   ./transfer_and_test.sh 192.168.64.2"
    echo ""
    exit 0
fi

# Build test_exec_stub
echo "🔨 Building Melvin tests..."
if [ -f "test_exec_stub.c" ]; then
    gcc -o test_exec_stub test_exec_stub.c -lm -std=c11 -Wall -Wextra 2>&1 | head -10
    echo "✅ test_exec_stub compiled"
else
    echo "⚠️  test_exec_stub.c not found"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Run tests:"
echo "  ./test_exec_stub"
echo ""
echo "Or from Mac, transfer files:"
echo "  cd ~/melvin_november/Melvin_november"
echo "  ./transfer_and_test.sh 192.168.64.2"
echo ""




