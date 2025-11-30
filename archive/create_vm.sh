#!/bin/bash
# Automated script to create Ubuntu VM for Jetson flashing

echo "=========================================="
echo "Creating Ubuntu VM for Jetson Flashing"
echo "=========================================="
echo ""

# Check if VirtualBox is installed
if ! command -v VBoxManage &> /dev/null; then
    echo "Error: VirtualBox not found!"
    echo "Install with: brew install --cask virtualbox"
    exit 1
fi

# Check if Ubuntu ISO exists
ISO_PATH="$HOME/Downloads/ubuntu-22.04.3-desktop-amd64.iso"
if [ ! -f "$ISO_PATH" ]; then
    echo "Error: Ubuntu ISO not found at $ISO_PATH"
    echo "Download from: https://releases.ubuntu.com/22.04/"
    exit 1
fi

VM_NAME="Ubuntu-Jetson-Flash"
VM_MEMORY=4096  # 4GB RAM
VM_DISK=50000   # 50GB disk

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
    --audio none \
    --boot1 dvd \
    --boot2 disk \
    --boot3 none \
    --boot4 none

# Create and attach storage
VBoxManage createhd --filename "$HOME/VirtualBox VMs/$VM_NAME/$VM_NAME.vdi" \
    --size $VM_DISK --format VDI

VBoxManage storagectl "$VM_NAME" --name "SATA Controller" \
    --add sata --controller IntelAHCI

VBoxManage storageattach "$VM_NAME" \
    --storagectl "SATA Controller" \
    --port 0 --device 0 \
    --type hdd \
    --medium "$HOME/VirtualBox VMs/$VM_NAME/$VM_NAME.vdi"

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
echo "Next steps:"
echo "1. Open VirtualBox"
echo "2. Start VM: $VM_NAME"
echo "3. Install Ubuntu 22.04"
echo "4. After installation, install SDK Manager:"
echo "   sudo apt update"
echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
echo "   sudo dpkg -i cuda-keyring_1.1-1_all.deb"
echo "   sudo apt-get update"
echo "   sudo apt-get -y install sdkmanager"
echo ""
echo "5. Put Jetson in Recovery Mode and connect via USB"
echo "6. In VM: Devices → USB → Select Jetson device"
echo "7. Run SDK Manager to flash JetPack 6.x"
echo ""
echo "See SETUP_JETSON_VM.md for detailed instructions"

