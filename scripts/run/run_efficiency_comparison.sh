#!/bin/bash
# Quick comparison test - focus on efficiency metrics

VM_USER="melvin"
VM_HOST="169.254.123.100"
VM_PASS="123456"
VM_DIR="~/melvin_november"

echo "=========================================="
echo "EFFICIENCY-AWARE FE: KEY METRICS TEST"
echo "=========================================="
echo ""

# Transfer files
echo "Transferring files..."
sshpass -p "$VM_PASS" rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
    melvin.c test_data_vs_computation.c test_exec_learning_simple.c \
    "$VM_USER@$VM_HOST:$VM_DIR/" > /dev/null 2>&1

echo ""
echo "TEST 1: Data vs Computation (Efficiency Impact)"
echo "----------------------------------------"
sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
    "cd $VM_DIR && \
     rm -f test_data_vs_computation.m && \
     gcc -o test_data_vs_computation test_data_vs_computation.c -lm -std=c11 2>&1 | grep error | head -2 && \
     timeout 120 ./test_data_vs_computation 2>&1 | grep -A 20 'ANALYSIS\|Pattern nodes\|EXEC nodes\|Output strength'"
echo ""

echo "TEST 2: EXEC Learning (Stability Check)"
echo "----------------------------------------"
sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
    "cd $VM_DIR && \
     rm -f test_exec_learning_simple.m && \
     gcc -o test_exec_learning_simple test_exec_learning_simple.c -lm -std=c11 2>&1 | grep error | head -2 && \
     timeout 60 ./test_exec_learning_simple 2>&1 | tail -15"
echo ""

echo "=========================================="
echo "KEY DIFFERENCES TO LOOK FOR:"
echo "=========================================="
echo "1. Pattern node count (should be lower with complexity penalty)"
echo "2. EXEC node stability (should be higher if efficient)"
echo "3. Overall graph size (should favor compact structures)"
echo "4. Efficiency scores (low FE per traffic = good)"
echo ""

