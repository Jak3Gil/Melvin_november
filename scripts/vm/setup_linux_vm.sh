#!/bin/bash
# Setup script for creating a Linux VM for Melvin development
# This script helps set up UTM and downloads a Linux ISO

set -e

echo "=========================================="
echo "Linux VM Setup for Melvin Development"
echo "=========================================="
echo ""

# Check if UTM is installed
if ! command -v utmctl &> /dev/null && ! [ -d "/Applications/UTM.app" ]; then
    echo "❌ UTM is not installed."
    echo "   Installing via Homebrew..."
    brew install --cask utm
    echo "✅ UTM installed. Please restart this script."
    exit 0
fi

echo "✅ UTM is installed"
echo ""

# Create directory for VM files
VM_DIR="$HOME/melvin_linux_vm"
mkdir -p "$VM_DIR"
cd "$VM_DIR"

echo "📁 VM directory: $VM_DIR"
echo ""

# Download Ubuntu Server ARM64 ISO if not already present
# Support both 22.04 and 24.04
ISO_FILE_2204="ubuntu-22.04-server-arm64.iso"
ISO_FILE_2404="ubuntu-24.04.3-live-server-arm64.iso"
ISO_FILE="${ISO_FILE_2404}"  # Default to 24.04
ISO_SIZE="1.2G"  # Approximate size

# Check if we need to download
NEED_DOWNLOAD=0
if [ ! -f "$ISO_FILE" ]; then
    NEED_DOWNLOAD=1
elif [ ! -s "$ISO_FILE" ]; then
    echo "⚠️  ISO file is empty, re-downloading..."
    NEED_DOWNLOAD=1
elif file "$ISO_FILE" 2>/dev/null | grep -q "HTML"; then
    echo "⚠️  ISO file appears to be HTML redirect, re-downloading..."
    rm -f "$ISO_FILE"
    NEED_DOWNLOAD=1
elif [ $(stat -f%z "$ISO_FILE" 2>/dev/null || stat -c%s "$ISO_FILE" 2>/dev/null) -lt 1000000 ]; then
    echo "⚠️  ISO file is too small (< 1MB), re-downloading..."
    rm -f "$ISO_FILE"
    NEED_DOWNLOAD=1
fi

if [ $NEED_DOWNLOAD -eq 1 ]; then
    echo "📥 Downloading Ubuntu Server 22.04 ARM64 ISO..."
    echo "   Size: ~$ISO_SIZE"
    echo "   This may take a while..."
    echo ""
    
    # Use wget or curl with proper redirect following
    # Try Ubuntu 24.04 first, fall back to 22.04 if needed
    ISO_URL_2404="https://cdimage.ubuntu.com/releases/24.04/release/ubuntu-24.04.3-live-server-arm64.iso"
    ISO_URL_2204="https://cdimage.ubuntu.com/releases/22.04/release/ubuntu-22.04.4-live-server-arm64.iso"
    
    DOWNLOAD_URL="$ISO_URL_2404"
    
    if command -v wget &> /dev/null; then
        wget --progress=bar:force -O "$ISO_FILE" \
            "$DOWNLOAD_URL" 2>&1 | \
            tail -1
    else
        # Use curl with location following and output to file
        curl -L --location-trusted --progress-bar \
            -o "$ISO_FILE" \
            "$DOWNLOAD_URL" || {
            echo "⚠️  Download from primary URL failed, trying alternative..."
            # Try 22.04 as fallback
            ISO_FILE="$ISO_FILE_2204"
            curl -L --location-trusted --progress-bar \
                -o "$ISO_FILE" \
                "$ISO_URL_2204" || {
                echo "❌ Download failed with curl"
                rm -f "$ISO_FILE"
            }
        }
    fi
    
    # Verify download
    if [ ! -f "$ISO_FILE" ] || [ ! -s "$ISO_FILE" ]; then
        echo "❌ Download failed - file is missing or empty"
        rm -f "$ISO_FILE"
        NEED_DOWNLOAD=1
    elif file "$ISO_FILE" 2>/dev/null | grep -q "HTML"; then
        echo "❌ Download failed - got HTML redirect instead of ISO"
        rm -f "$ISO_FILE"
        NEED_DOWNLOAD=1
    elif [ $(stat -f%z "$ISO_FILE" 2>/dev/null || stat -c%s "$ISO_FILE" 2>/dev/null) -lt 1000000 ]; then
        echo "❌ Download failed - file too small"
        rm -f "$ISO_FILE"
        NEED_DOWNLOAD=1
    else
        ISO_SIZE_ACTUAL=$(du -h "$ISO_FILE" | cut -f1)
        echo "✅ Download complete: $ISO_FILE ($ISO_SIZE_ACTUAL)"
        NEED_DOWNLOAD=0
    fi
    
    if [ $NEED_DOWNLOAD -eq 1 ]; then
        echo ""
        echo "❌ Automatic download failed."
        echo ""
        echo "Please manually download Ubuntu Server 24.04 ARM64:"
        echo "   1. Visit: https://ubuntu.com/download/server/arm"
        echo "   2. Download: Ubuntu Server 24.04 LTS (ARM64)"
        echo "   3. Or use direct link:"
        echo "      https://cdimage.ubuntu.com/releases/24.04/release/ubuntu-24.04.3-live-server-arm64.iso"
        echo "   4. Alternative (22.04 LTS):"
        echo "      https://cdimage.ubuntu.com/releases/22.04/release/ubuntu-22.04.4-live-server-arm64.iso"
        echo ""
        echo "   4. Save it as: $VM_DIR/$ISO_FILE"
        echo ""
        echo "Then run this script again or proceed manually."
        exit 1
    fi
else
    ISO_SIZE_ACTUAL=$(du -h "$ISO_FILE" 2>/dev/null | cut -f1)
    echo "✅ ISO file already exists: $ISO_FILE ($ISO_SIZE_ACTUAL)"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Open UTM (from Applications)"
echo ""
echo "2. Create a new VM:"
echo "   - Click '+' or 'New'"
echo "   - Choose 'Virtualize' (not Emulate)"
echo "   - Select 'Linux'"
echo "   - Choose 'Browse...' and select:"
echo "     $VM_DIR/$ISO_FILE"
echo ""
echo "   Note: We'll use Ubuntu 24.04.3. If you prefer 22.04, look for:"
echo "     $VM_DIR/$ISO_FILE_2204"
echo ""
echo "3. Configure the VM:"
echo "   - RAM: 4 GB (4096 MB) minimum, 8 GB recommended"
echo "   - CPU cores: 4+ recommended"
echo "   - Disk: 20 GB minimum"
echo "   - Network: Shared Network (NAT)"
echo ""
echo "4. Install Ubuntu Server:"
echo "   - Boot the VM and follow the installation wizard"
echo "   - Create a user account (remember the password)"
echo "   - Install OpenSSH server when prompted"
echo ""
echo "5. After installation:"
echo "   - Run: ./transfer_melvin_to_vm.sh"
echo "   - Or manually copy the Melvin codebase"
echo ""
echo "=========================================="
echo "VM files location: $VM_DIR"
echo "=========================================="

