#!/bin/bash

# Transfer and run theoretical capability tests on Linux VM
# This is where RWX permissions actually work

set -e

VM_HOST="${1:-192.168.64.2}"
VM_USER="${2:-ubuntu}"
VM_SSH_PORT="${3:-22}"

echo "=========================================="
echo "Running Theoretical Tests on Linux VM"
echo "=========================================="
echo ""
echo "VM: $VM_USER@$VM_HOST:$VM_SSH_PORT"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ! ssh -p "$VM_SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" "echo '✓ Connected'" 2>/dev/null; then
    echo "❌ Cannot connect to VM via SSH"
    echo ""
    echo "Usage: $0 [vm_ip] [vm_user] [ssh_port]"
    echo "Example: $0 192.168.64.2 ubuntu"
    echo ""
    echo "To find VM IP (inside VM):"
    echo "  ip addr show | grep 'inet ' | grep -v '127.0.0.1'"
    exit 1
fi

echo "✅ SSH connection successful"
echo ""

# Transfer test files
echo "📤 Transferring theoretical test files..."
rsync -avz --progress -e "ssh -p $VM_SSH_PORT -o StrictHostKeyChecking=no" \
    --include='test_self_modify.c' \
    --include='test_code_evolution.c' \
    --include='test_auto_exec.c' \
    --include='test_meta_learning.c' \
    --include='test_emergent_algo.c' \
    --include='run_theoretical_tests.sh' \
    --include='melvin.c' \
    --include='melvin.h' \
    --exclude='*' \
    "$(pwd)/" "$VM_USER@$VM_HOST:~/melvin_november/" 2>&1 | tail -5

echo ""
echo "✅ Transfer complete"
echo ""

# Run tests on VM
echo "🧪 Running theoretical tests on Linux VM..."
echo ""

ssh -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" "
    cd ~/melvin_november
    chmod +x run_theoretical_tests.sh
    ./run_theoretical_tests.sh
"

echo ""
echo "=========================================="
echo "Tests Complete!"
echo "=========================================="
echo ""
echo "Results shown above. Check which theoretical capabilities work on Linux."
echo ""

