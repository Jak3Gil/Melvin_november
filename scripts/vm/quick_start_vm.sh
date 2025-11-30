#!/bin/bash
# Quick start script - creates VM and sets up Melvin for testing

set -e

echo "=========================================="
echo "Melvin VM Quick Start"
echo "=========================================="
echo ""

VM_DIR="$HOME/melvin_linux_vm"
ISO_FILE="$VM_DIR/ubuntu-24.04.3-live-server-arm64.iso"

# Check ISO exists
if [ ! -f "$ISO_FILE" ]; then
    echo "❌ ISO file not found: $ISO_FILE"
    echo "   Please download it first or run: ./check_iso.sh"
    exit 1
fi

echo "✅ ISO found: $ISO_FILE"
echo ""

# Check if UTM is installed
if [ ! -d "/Applications/UTM.app" ]; then
    echo "❌ UTM is not installed."
    echo "   Installing via Homebrew..."
    brew install --cask utm
fi

echo "✅ UTM is installed"
echo ""

# Open UTM
echo "🚀 Opening UTM..."
open -a UTM

echo ""
echo "=========================================="
echo "Next Steps in UTM:"
echo "=========================================="
echo ""
echo "1. Click '+' to create new VM"
echo "2. Choose 'Virtualize' → 'Linux'"
echo "3. Browse for ISO: $ISO_FILE"
echo "4. Configure:"
echo "   - RAM: 4-8 GB"
echo "   - CPU: 4 cores"
echo "   - Disk: 20-40 GB"
echo "   - Network: Shared (NAT)"
echo ""
echo "5. Install Ubuntu:"
echo "   - Check 'Install OpenSSH server'"
echo "   - Username: ubuntu"
echo ""
echo "6. After installation, get VM IP and run:"
echo "   ./transfer_and_test.sh <vm_ip>"
echo ""
echo "=========================================="




