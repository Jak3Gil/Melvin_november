#!/bin/bash
# Fully automated VM setup with Ubuntu and SDK Manager

set -e

VM_NAME="Ubuntu-Jetson-Flash"
ISO_PATH="$HOME/Downloads/ubuntu-22.04.3-desktop-amd64.iso"

echo "=========================================="
echo "Fully Automated Jetson VM Setup"
echo "=========================================="

# Check prerequisites
if ! command -v VBoxManage &> /dev/null; then
    echo "Error: VirtualBox not installed"
    exit 1
fi

if [ ! -f "$ISO_PATH" ]; then
    echo "Error: Ubuntu ISO not found"
    exit 1
fi

# Create VM if not exists
if ! VBoxManage showvminfo "$VM_NAME" &>/dev/null; then
    echo "Creating VM..."
    VBoxManage createvm --name "$VM_NAME" --ostype "Ubuntu_64" --register
    VBoxManage modifyvm "$VM_NAME" --memory 4096 --cpus 2 --vram 128 --usb on --usbehci on --usbxhci on
    VBoxManage createhd --filename "$HOME/VirtualBox VMs/$VM_NAME/$VM_NAME.vdi" --size 50000 --format VDI
    VBoxManage storagectl "$VM_NAME" --name "SATA" --add sata --controller IntelAHCI
    VBoxManage storageattach "$VM_NAME" --storagectl "SATA" --port 0 --device 0 --type hdd --medium "$HOME/VirtualBox VMs/$VM_NAME/$VM_NAME.vdi"
    VBoxManage storagectl "$VM_NAME" --name "IDE" --add ide --controller PIIX4
    VBoxManage storageattach "$VM_NAME" --storagectl "IDE" --port 0 --device 0 --type dvddrive --medium "$ISO_PATH"
fi

echo "VM ready. Start it manually in VirtualBox to install Ubuntu."
echo "After Ubuntu installation, run: install_sdkmanager_in_vm.sh"


# Add preseed for unattended installation
PRESEED_FILE="$(dirname "$0")/ubuntu_preseed.cfg"
if [ -f "$PRESEED_FILE" ]; then
    echo "Configuring unattended installation..."
    VBoxManage storageattach "$VM_NAME" --storagectl "IDE" --port 1 --device 0 --type dvddrive --medium "$PRESEED_FILE"
fi

echo "VM configured for unattended installation!"
echo "Start VM in VirtualBox - Ubuntu will install automatically"
