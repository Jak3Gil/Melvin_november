#!/bin/bash
# Run RWX automatic test on VM

set -e

VM_USER="${2:-melvin}"
VM_HOST="${1:-192.168.64.2}"

echo "=========================================="
echo "RWX Automatic Test Runner"
echo "=========================================="
echo ""

# Transfer test file
echo "Transferring test_rwx_automatic.c..."
scp test_rwx_automatic.c melvin.c melvin.h melvin@192.168.64.2:~/melvin_november/ 2>&1 | tail -3

echo ""
echo "Compiling and running on VM..."
ssh melvin@192.168.64.2 "
    cd ~/melvin_november
    gcc -o test_rwx_automatic test_rwx_automatic.c -lm -std=c11 -Wall -Wextra -O0
    ./test_rwx_automatic
"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓✓✓ RWX TEST PASSED ✓✓✓"
else
    echo "✗✗✗ RWX TEST FAILED ✗✗✗"
fi

exit $EXIT_CODE

