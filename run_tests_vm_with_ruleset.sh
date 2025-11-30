#!/bin/bash

# Run theoretical tests on Linux VM with Master Ruleset validation
# This tests how the Master Ruleset enforcement affects behavior

set -e

VM_HOST="${1:-192.168.64.2}"
VM_USER="${2:-melvin}"
VM_PASS="${3:-1234}"

echo "=========================================="
echo "Running Tests with Master Ruleset Validation"
echo "=========================================="
echo ""
echo "VM: $VM_USER@$VM_HOST"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ! sshpass -p "$VM_PASS" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" "echo '✓ Connected'" 2>/dev/null; then
    echo "❌ Cannot connect to VM via SSH"
    echo ""
    echo "Make sure:"
    echo "  1. VM is running"
    echo "  2. SSH is enabled in VM"
    echo "  3. IP address is correct"
    echo ""
    echo "To find VM IP (inside VM):"
    echo "  ip addr show | grep 'inet ' | grep -v '127.0.0.1'"
    echo ""
    echo "Usage: $0 [vm_ip] [vm_user] [vm_password]"
    exit 1
fi

echo "✅ SSH connection successful"
echo ""

# Transfer updated files
echo "📤 Transferring updated melvin.c with Master Ruleset..."
sshpass -p "$VM_PASS" rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no" \
    melvin.c melvin.h \
    "$VM_USER@$VM_HOST:~/melvin_november/" 2>&1 | tail -5

echo ""
echo "✅ Transfer complete"
echo ""

# Run tests
echo "🧪 Running theoretical tests with Master Ruleset validation..."
echo ""

sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" "
    cd ~/melvin_november
    chmod +x run_theoretical_tests.sh
    ./run_theoretical_tests.sh
"

echo ""
echo "=========================================="
echo "Tests Complete!"
echo "=========================================="
echo ""
echo "Check results above. Master Ruleset validation will:"
echo "  - Catch violations early (fail-fast)"
echo "  - Log which section was violated"
echo "  - Reset invalid states to safe defaults"
echo ""


