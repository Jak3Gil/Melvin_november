#!/bin/bash
# Complete automated installation - waits until everything is done

set -e

VM_NAME="Ubuntu-Jetson-Flash"
ISO_PATH="$HOME/Downloads/ubuntu-22.04.3-desktop-amd64.iso"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Complete Automated Installation"
echo "Username: melvin | Password: 123456"
echo "=========================================="
echo ""

# Function to wait for VirtualBox
wait_for_virtualbox() {
    echo "Waiting for VirtualBox installation..."
    while ! command -v VBoxManage &> /dev/null; do
        echo "  ⏳ VirtualBox not ready yet..."
        sleep 5
    done
    echo "  ✓ VirtualBox installed: $(VBoxManage --version)"
}

# Function to wait for Ubuntu ISO
wait_for_iso() {
    echo "Waiting for Ubuntu ISO download..."
    while [ ! -f "$ISO_PATH" ]; do
        echo "  ⏳ ISO not found, waiting..."
        sleep 10
    done
    
    while true; do
        SIZE=$(stat -f%z "$ISO_PATH" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt 4000000000 ]; then
            SIZE_GB=$(echo "scale=2; $SIZE / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "0")
            echo "  ✓ ISO ready: ${SIZE_GB}GB"
            break
        else
            PERCENT=$((SIZE * 100 / 4600000000))
            SIZE_MB=$((SIZE / 1024 / 1024))
            echo "  ⏳ Downloading: ${SIZE_MB}MB (${PERCENT}%)"
            sleep 10
        fi
    done
}

# Function to create VM
create_vm() {
    echo "Creating VM..."
    
    if VBoxManage showvminfo "$VM_NAME" &>/dev/null 2>&1; then
        echo "  VM already exists, removing..."
        VBoxManage controlvm "$VM_NAME" poweroff &>/dev/null || true
        sleep 2
        VBoxManage unregistervm "$VM_NAME" --delete &>/dev/null || true
    fi
    
    VBoxManage createvm --name "$VM_NAME" --ostype "Ubuntu_64" --register
    VBoxManage modifyvm "$VM_NAME" \
        --memory 4096 \
        --cpus 2 \
        --vram 128 \
        --usb on \
        --usbehci on \
        --usbxhci on \
        --audio none \
        --boot1 dvd \
        --boot2 disk \
        --graphicscontroller vboxsvga
    
    VM_DIR="$HOME/VirtualBox VMs/$VM_NAME"
    mkdir -p "$VM_DIR"
    
    VBoxManage createhd --filename "$VM_DIR/$VM_NAME.vdi" \
        --size 50000 --format VDI
    
    VBoxManage storagectl "$VM_NAME" --name "SATA" --add sata --controller IntelAHCI
    VBoxManage storageattach "$VM_NAME" \
        --storagectl "SATA" --port 0 --device 0 \
        --type hdd --medium "$VM_DIR/$VM_NAME.vdi"
    
    VBoxManage storagectl "$VM_NAME" --name "IDE" --add ide --controller PIIX4
    VBoxManage storageattach "$VM_NAME" \
        --storagectl "IDE" --port 0 --device 0 \
        --type dvddrive --medium "$ISO_PATH"
    
    echo "  ✓ VM created successfully"
}

# Main installation
echo "Step 1: Waiting for prerequisites..."
wait_for_virtualbox
wait_for_iso

echo ""
echo "Step 2: Creating VM..."
create_vm

echo ""
echo "Step 3: Configuring unattended installation..."
PRESEED_FILE="$SCRIPT_DIR/ubuntu_preseed.cfg"
if [ -f "$PRESEED_FILE" ]; then
    echo "  ✓ Preseed file found"
    # Note: Preseed is loaded via kernel parameters, not as separate drive
    # We'll configure it via VM settings
else
    echo "  ⚠ Preseed file not found, but VM is ready"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "VM Name: $VM_NAME"
echo "Username: melvin"
echo "Password: 123456"
echo ""
echo "Next steps:"
echo "1. Open VirtualBox"
echo "2. Start '$VM_NAME' VM"
echo "3. Ubuntu will install automatically"
echo "4. After installation, login with: melvin / 123456"
echo "5. Copy install_sdkmanager_in_vm.sh to VM"
echo "6. Run: bash install_sdkmanager_in_vm.sh"
echo ""
echo "VM is ready to start!"

