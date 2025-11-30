#!/bin/bash
# Transfer Melvin to VM and run initial tests

set -e

VM_USER="${2:-ubuntu}"
VM_HOST="${1}"
VM_SSH_PORT="${3:-22}"

if [ -z "$VM_HOST" ]; then
    echo "=========================================="
    echo "Transfer Melvin to VM and Run Tests"
    echo "=========================================="
    echo ""
    echo "Usage: $0 <vm_ip> [vm_user] [ssh_port]"
    echo ""
    echo "Example:"
    echo "  $0 192.168.64.3 ubuntu"
    echo ""
    echo "To find VM IP (inside VM):"
    echo "  ip addr show | grep 'inet ' | grep -v '127.0.0.1'"
    exit 1
fi

MELVIN_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Setting up Melvin on Linux VM"
echo "=========================================="
echo ""
echo "VM: $VM_USER@$VM_HOST:$VM_SSH_PORT"
echo "Melvin: $MELVIN_DIR"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ! ssh -p "$VM_SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" "echo '✓ Connected'" 2>/dev/null; then
    echo "❌ Cannot connect to VM via SSH"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Make sure VM is running and Ubuntu is installed"
    echo "  2. Check SSH is running: sudo systemctl status ssh (in VM)"
    echo "  3. Verify IP address"
    exit 1
fi

echo "✅ SSH connection successful"
echo ""

# Install build tools
echo "📦 Installing build tools on VM..."
ssh -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" "
    sudo apt update -qq
    sudo apt install -y build-essential gcc make 2>&1 | tail -3
    echo '✅ Build tools ready'
" || echo "⚠️  Build tools install had issues (continuing anyway)"

echo ""

# Transfer files
echo "📤 Transferring Melvin codebase..."
rsync -avz --progress -e "ssh -p $VM_SSH_PORT -o StrictHostKeyChecking=no" \
    --exclude='*.m' \
    --exclude='*.o' \
    --exclude='test_*' \
    --exclude='archive/' \
    --exclude='.git/' \
    --exclude='node_modules/' \
    "$MELVIN_DIR/" "$VM_USER@$VM_HOST:~/melvin_november/" 2>&1 | tail -5 || {
    echo "⚠️  rsync failed, trying scp..."
    ssh -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" "mkdir -p ~/melvin_november"
    scp -r -P "$VM_SSH_PORT" -o StrictHostKeyChecking=no \
        "$MELVIN_DIR/melvin.c" \
        "$MELVIN_DIR/melvin.h" \
        "$MELVIN_DIR/test_*.c" \
        "$MELVIN_DIR/*.sh" \
        "$VM_USER@$VM_HOST:~/melvin_november/" 2>&1 | tail -3
}

echo ""
echo "✅ Transfer complete"
echo ""

# Build and test
echo "🔨 Building Melvin on Linux..."
ssh -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" "
    cd ~/melvin_november
    echo 'Compiling test_exec_stub...'
    gcc -o test_exec_stub test_exec_stub.c -lm -std=c11 -Wall -Wextra 2>&1 | head -10
    if [ -f test_exec_stub ]; then
        echo '✅ Build successful'
        echo ''
        echo '🧪 Running EXEC stub test...'
        echo ''
        ./test_exec_stub 2>&1 | tail -20
    else
        echo '❌ Build failed'
        exit 1
    fi
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "SSH into VM:"
echo "  ssh -p $VM_SSH_PORT $VM_USER@$VM_HOST"
echo ""
echo "In the VM:"
echo "  cd ~/melvin_november"
echo "  ./build_and_test.sh"
echo ""
echo "Or run specific tests:"
echo "  gcc -o test_exec_stub test_exec_stub.c -lm -std=c11 && ./test_exec_stub"
echo "  gcc -o test_run_20min test_run_20min.c -lm -std=c11 && ./test_run_20min"
echo ""




