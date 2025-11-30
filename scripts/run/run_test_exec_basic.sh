#!/bin/bash
# Build and run test_exec_basic on Linux VM

set -e

VM_USER="${2:-melvin}"
VM_HOST="${1:-192.168.64.2}"

echo "=========================================="
echo "Building and Running test_exec_basic"
echo "=========================================="
echo ""

# Transfer test file
echo "Transferring test_exec_basic.c..."
scp test_exec_basic.c melvin@${VM_HOST}:~/melvin_november/ 2>&1 | tail -2

echo ""
echo "Compiling and running on VM..."
ssh melvin@${VM_HOST} "
    cd ~/melvin_november
    echo 'Compiling test_exec_basic.c...'
    gcc -std=c11 -Wall -Wextra -O0 -o test_exec_basic test_exec_basic.c
    echo ''
    echo 'Running test_exec_basic...'
    echo ''
    ./test_exec_basic
"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓✓✓ test_exec_basic: PASSED ✓✓✓"
else
    echo "✗✗✗ test_exec_basic: FAILED ✗✗✗"
fi

exit $EXIT_CODE

