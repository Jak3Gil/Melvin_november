#!/bin/bash
# Setup script for Jetson Orin AGX
# Installs dependencies and prepares the system

set -e

echo "=========================================="
echo "Jetson Orin AGX Setup for Melvin"
echo "=========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices"
    echo "Continuing anyway..."
fi

# Update package list
echo ""
echo "Updating package list..."
sudo apt-get update

# Install build essentials
echo ""
echo "Installing build tools..."
sudo apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    git \
    curl \
    wget

# Install development libraries
echo ""
echo "Installing development libraries..."
sudo apt-get install -y \
    libdl-dev \
    libpthread-stubs0-dev

# Install optional tools
echo ""
echo "Installing optional tools..."
sudo apt-get install -y \
    htop \
    tmux \
    vim \
    net-tools

# Check for curl (needed for mc_api plugin)
echo ""
echo "Checking for curl..."
if command -v curl >/dev/null 2>&1; then
    echo "  ✓ curl is installed"
else
    echo "  Installing curl..."
    sudo apt-get install -y curl
fi

# Check for git (needed for mc_git plugin)
echo ""
echo "Checking for git..."
if command -v git >/dev/null 2>&1; then
    echo "  ✓ git is installed"
else
    echo "  Installing git..."
    sudo apt-get install -y git
fi

# Check for Ollama (optional, for LLM API)
echo ""
echo "Checking for Ollama (optional)..."
if command -v ollama >/dev/null 2>&1; then
    echo "  ✓ Ollama is installed"
    echo "  To start: ollama serve"
else
    echo "  Ollama not found. Install from: https://ollama.ai"
    echo "  Or: curl -fsSL https://ollama.ai/install.sh | sh"
fi

# Set up memory limits (64GB RAM - no special limits needed)
echo ""
echo "System memory:"
free -h

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p plugins
mkdir -p scaffolds
mkdir -p ingested_repos
mkdir -p data
mkdir -p corpus

# Set permissions
chmod +x build_jetson.sh
chmod +x run_jetson.sh 2>/dev/null || true

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: ./build_jetson.sh"
echo "  2. Initialize brain: ./init_melvin.sh (if exists)"
echo "  3. Run: ./melvin"
echo ""

