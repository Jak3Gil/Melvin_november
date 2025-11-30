#!/bin/bash
# Transfer and run universal stress test on Linux VM

set -e

# Default VM connection (from memory: 169.254.123.100)
VM_USER="${2:-melvin}"
VM_HOST="${1:-192.168.64.2}"
VM_SSH_PORT="${3:-22}"

MELVIN_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Run Universal Stress Test on Linux VM"
echo "=========================================="
echo ""
echo "VM: $VM_USER@$VM_HOST:$VM_SSH_PORT"
echo "Source: $MELVIN_DIR"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ! ssh -p "$VM_SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" "echo '✓ Connected'" 2>/dev/null; then
    echo "❌ Cannot connect to VM via SSH"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Make sure VM is running"
    echo "  2. Check SSH is running: sudo systemctl status ssh (in VM)"
    echo "  3. Verify IP address: $VM_HOST"
    echo "  4. Try: ssh -p $VM_SSH_PORT $VM_USER@$VM_HOST"
    echo ""
    echo "Usage: $0 [vm_ip] [vm_user] [ssh_port]"
    echo "Example: $0 192.168.64.2 ubuntu 22"
    exit 1
fi

echo "✅ SSH connection successful"
echo ""

# Transfer the new test file
echo "📤 Transferring test_universal_stress.c and runner script..."
rsync -avz --progress -e "ssh -p $VM_SSH_PORT -o StrictHostKeyChecking=no" \
    "$MELVIN_DIR/test_universal_stress.c" \
    "$MELVIN_DIR/run_universal_stress_test.sh" \
    "$MELVIN_DIR/melvin.c" \
    "$MELVIN_DIR/melvin.h" \
    "$VM_USER@$VM_HOST:~/melvin_november/" 2>&1 | tail -5 || {
    echo "⚠️  rsync failed, trying scp..."
    scp -P "$VM_SSH_PORT" -o StrictHostKeyChecking=no \
        "$MELVIN_DIR/test_universal_stress.c" \
        "$MELVIN_DIR/run_universal_stress_test.sh" \
        "$MELVIN_DIR/melvin.c" \
        "$MELVIN_DIR/melvin.h" \
        "$VM_USER@$VM_HOST:~/melvin_november/" 2>&1 | tail -3
}

echo ""
echo "✅ Transfer complete"
echo ""

# Make runner script executable and run the test
echo "🔨 Compiling and running universal stress test on VM..."
echo ""
ssh -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" "
    cd ~/melvin_november
    chmod +x run_universal_stress_test.sh 2>/dev/null || true
    
    echo 'Compiling test_universal_stress.c...'
    gcc -o test_universal_stress test_universal_stress.c -lm -std=c11 -Wall -Wextra -O2
    
    if [ \$? -ne 0 ]; then
        echo '❌ Compilation failed'
        exit 1
    fi
    
    echo '✅ Compilation successful'
    echo ''
    echo '=========================================='
    echo 'Starting Universal Stress Test'
    echo '=========================================='
    echo ''
    
    ./test_universal_stress
    
    EXIT_CODE=\$?
    echo ''
    echo '=========================================='
    if [ \$EXIT_CODE -eq 0 ]; then
        echo '✓✓✓ TEST COMPLETED SUCCESSFULLY ✓✓✓'
    else
        echo '✗✗✗ TEST FAILED OR WAS INTERRUPTED ✗✗✗'
    fi
    echo '=========================================='
    
    exit \$EXIT_CODE
"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓✓✓ TEST COMPLETED ON VM ✓✓✓"
else
    echo "✗✗✗ TEST FAILED ON VM ✗✗✗"
fi
echo "=========================================="
echo ""
echo "To SSH into VM and check results:"
echo "  ssh -p $VM_SSH_PORT $VM_USER@$VM_HOST"
echo "  cd ~/melvin_november"
echo "  ls -lh test_universal_stress*"
echo ""

exit $EXIT_CODE

