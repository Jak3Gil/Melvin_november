#!/bin/bash
# Transfer Melvin codebase to Linux VM
# Usage: ./transfer_melvin_to_vm.sh [vm_ip_or_hostname] [vm_user]

set -e

VM_USER="${2:-ubuntu}"  # Default username
VM_HOST="${1}"
VM_SSH_PORT="${3:-22}"

if [ -z "$VM_HOST" ]; then
    echo "=========================================="
    echo "Transfer Melvin to Linux VM"
    echo "=========================================="
    echo ""
    echo "Usage: $0 <vm_ip_or_hostname> [vm_user] [ssh_port]"
    echo ""
    echo "Example:"
    echo "  $0 192.168.64.2 ubuntu"
    echo ""
    echo "To find your VM's IP address:"
    echo "  1. SSH into the VM"
    echo "  2. Run: ip addr show | grep 'inet '"
    echo ""
    exit 1
fi

MELVIN_DIR="$(cd "$(dirname "$0")" && pwd)"
VM_TARGET_DIR="~/melvin_november"

echo "=========================================="
echo "Transferring Melvin to Linux VM"
echo "=========================================="
echo ""
echo "Source: $MELVIN_DIR"
echo "Target: $VM_USER@$VM_HOST:$VM_TARGET_DIR"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ! ssh -p "$VM_SSH_PORT" -o ConnectTimeout=5 "$VM_USER@$VM_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
    echo "❌ Cannot connect to VM via SSH"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Make sure the VM is running"
    echo "  2. Check that SSH is installed on the VM:"
    echo "     sudo apt update && sudo apt install -y openssh-server"
    echo "  3. Verify the IP address/hostname"
    echo "  4. Check if you need to use a different SSH port"
    echo ""
    exit 1
fi

echo "✅ SSH connection successful"
echo ""

# Install build tools on VM if needed
echo "Installing build tools on VM..."
ssh -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" "sudo apt update && sudo apt install -y build-essential git 2>&1 | tail -5" || {
    echo "⚠️  Warning: Could not install build tools automatically"
}

echo ""

# Create target directory on VM
echo "Creating target directory on VM..."
ssh -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" "mkdir -p $VM_TARGET_DIR"

# Transfer files
echo "Transferring Melvin codebase..."
echo "   (This may take a moment...)"

rsync -avz -e "ssh -p $VM_SSH_PORT" \
    --exclude='*.m' \
    --exclude='*.o' \
    --exclude='test_*' \
    --exclude='archive/' \
    --exclude='.git/' \
    "$MELVIN_DIR/" "$VM_USER@$VM_HOST:$VM_TARGET_DIR/" || {
    echo ""
    echo "⚠️  rsync not available, trying scp..."
    scp -r -P "$VM_SSH_PORT" \
        "$MELVIN_DIR/melvin.c" \
        "$MELVIN_DIR/melvin.h" \
        "$MELVIN_DIR/test_*.c" \
        "$VM_USER@$VM_HOST:$VM_TARGET_DIR/"
}

echo ""
echo "✅ Transfer complete!"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "SSH into the VM:"
echo "  ssh -p $VM_SSH_PORT $VM_USER@$VM_HOST"
echo ""
echo "Then in the VM:"
echo "  cd ~/melvin_november"
echo "  ./build_and_test.sh"
echo ""
echo "Or manually:"
echo "  gcc -o test_exec_stub test_exec_stub.c -lm -std=c11"
echo "  ./test_exec_stub"
echo ""

