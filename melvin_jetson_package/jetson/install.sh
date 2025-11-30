#!/bin/bash
#
# install.sh - Install Melvin on Jetson
#

set -e

echo "=========================================="
echo "Installing Melvin on Jetson"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠ Warning: This doesn't appear to be a Jetson device"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo "1. Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libasound2-dev \
    python3 \
    python3-pip \
    curl \
    jq \
    binutils

# Install tools (if script exists)
if [ -f tools/install_tools_jetson.sh ]; then
    echo ""
    echo "2. Installing AI tools..."
    bash tools/install_tools_jetson.sh
else
    echo "  ⚠ Tools installation script not found"
fi

# Compile Melvin
echo ""
echo "3. Compiling Melvin..."
make clean
make

# Create data directory
echo ""
echo "4. Creating data directory..."
mkdir -p ~/melvin_data
mkdir -p ~/melvin_data/brains

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Binaries:"
echo "  ./melvin_hardware_runner  - Run with hardware (mic, speaker, camera)"
echo "  ./melvin_run_continuous   - Run without hardware"
echo ""
echo "Data directory: ~/melvin_data/brains/"
echo ""
