#!/bin/bash
# Fully automated VM setup for Jetson flashing

set -e

echo "=========================================="
echo "Automatic VM Setup for Jetson Flashing"
echo "=========================================="
echo ""

# Check VirtualBox
if ! command -v VBoxManage &> /dev/null; then
    echo "Error: VirtualBox not installed!"
    echo "Please run: brew install --cask virtualbox"
    echo "Then enter your password when prompted"
    exit 1
fi

echo "✓ VirtualBox found: $(VBoxManage --version)"

# Check Ubuntu ISO
ISO_PATH="$HOME/Downloads/ubuntu-22.04.3-desktop-amd64.iso"
if [ ! -f "$ISO_PATH" ]; then
    echo "Error: Ubuntu ISO not found at $ISO_PATH"
    echo "Downloading now..."
    cd ~/Downloads
    curl -L -o ubuntu-22.04.3-desktop-amd64.iso \
        https://releases.ubuntu.com/22.04/ubuntu-22.04.3-desktop-amd64.iso
    if [ ! -f "$ISO_PATH" ]; then
        echo "Download failed. Please download manually:"
        echo "https://releases.ubuntu.com/22.04/ubuntu-22.04.3-desktop-amd64.iso"
        exit 1
    fi
fi

ISO_SIZE=$(stat -f%z "$ISO_PATH" 2>/dev/null || stat -c%s "$ISO_PATH" 2>/dev/null || echo 0)
if [ "$ISO_SIZE" -lt 4000000000 ]; then
    echo "Warning: ISO file seems incomplete ($ISO_SIZE bytes)"
    echo "Expected ~4.6GB. Please check download."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✓ Ubuntu ISO found: $(ls -lh "$ISO_PATH" | awk '{print $5}')"

VM_NAME="Ubuntu-Jetson-Flash"
VM_MEMORY=4096
VM_DISK=50000

# Check if VM already exists
if VBoxManage showvminfo "$VM_NAME" &>/dev/null; then
    echo "VM '$VM_NAME' already exists."
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing VM..."
        VBoxManage controlvm "$VM_NAME" poweroff &>/dev/null || true
        sleep 2
        VBoxManage unregistervm "$VM_NAME" --delete &>/dev/null || true
    else
        echo "Using existing VM."
        exit 0
    fi
fi

echo ""
echo "Creating VM: $VM_NAME"
echo "Memory: ${VM_MEMORY}MB"
echo "Disk: ${VM_DISK}MB"
echo ""

# Create VM
VBoxManage createvm --name "$VM_NAME" --ostype "Ubuntu_64" --register

# Configure VM
VBoxManage modifyvm "$VM_NAME" \
    --memory $VM_MEMORY \
    --cpus 2 \
    --vram 128 \
    --usb on \
    --usbehci on \
    --usbxhci on \
    --audio none \
    --boot1 dvd \
    --boot2 disk \
    --boot3 none \
    --boot4 none \
    --graphicscontroller vboxsvga

# Create and attach storage
VM_DIR="$HOME/VirtualBox VMs/$VM_NAME"
mkdir -p "$VM_DIR"

VBoxManage createhd --filename "$VM_DIR/$VM_NAME.vdi" \
    --size $VM_DISK --format VDI

VBoxManage storagectl "$VM_NAME" --name "SATA Controller" \
    --add sata --controller IntelAHCI

VBoxManage storageattach "$VM_NAME" \
    --storagectl "SATA Controller" \
    --port 0 --device 0 \
    --type hdd \
    --medium "$VM_DIR/$VM_NAME.vdi"

# Attach Ubuntu ISO
VBoxManage storagectl "$VM_NAME" --name "IDE Controller" \
    --add ide --controller PIIX4

VBoxManage storageattach "$VM_NAME" \
    --storagectl "IDE Controller" \
    --port 0 --device 0 \
    --type dvddrive \
    --medium "$ISO_PATH"

echo ""
echo "=========================================="
echo "VM Created Successfully!"
echo "=========================================="
echo ""
echo "VM Name: $VM_NAME"
echo "Status: Ready to install Ubuntu"
echo ""
echo "Next steps:"
echo "1. Open VirtualBox"
echo "2. Start VM: $VM_NAME"
echo "3. Install Ubuntu 22.04 (follow on-screen instructions)"
echo "4. After Ubuntu installation, run in VM:"
echo ""
echo "   sudo apt update"
echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
echo "   sudo dpkg -i cuda-keyring_1.1-1_all.deb"
echo "   sudo apt-get update"
echo "   sudo apt-get -y install sdkmanager"
echo ""
echo "5. Put Jetson in Recovery Mode and connect via USB"
echo "6. In VirtualBox: Devices → USB → Select Jetson"
echo "7. Run: sdkmanager"
echo ""
echo "See QUICK_START_VM.md for detailed instructions"

