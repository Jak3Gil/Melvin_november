#!/bin/bash
# Run all tests with efficiency-aware free-energy implementation
# Compare results to see impact of complexity-aware stability

set -e

VM_USER="melvin"
VM_HOST="169.254.123.100"
VM_PASS="123456"
VM_DIR="~/melvin_november"

echo "=========================================="
echo "RUNNING ALL TESTS WITH EFFICIENCY-AWARE FE"
echo "=========================================="
echo ""

# Transfer melvin.c and all test files
echo "Step 1: Transferring files to VM..."
sshpass -p "$VM_PASS" rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no" \
    melvin.c \
    test_*.c \
    "$VM_USER@$VM_HOST:$VM_DIR/" 2>&1 | tail -3

echo ""
echo "Step 2: Compiling and running tests..."
echo ""

# Test 1: Data vs Computation
echo "----------------------------------------"
echo "TEST 1: Data vs Computation"
echo "----------------------------------------"
sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
    "cd $VM_DIR && \
     rm -f test_data_vs_computation.m && \
     gcc -o test_data_vs_computation test_data_vs_computation.c -lm -std=c11 -Wall 2>&1 | grep -E 'error|Error' | head -3 && \
     timeout 300 ./test_data_vs_computation 2>&1 | tail -30"
echo ""

# Test 2: EXEC Learning Simple
echo "----------------------------------------"
echo "TEST 2: EXEC Learning Simple"
echo "----------------------------------------"
sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
    "cd $VM_DIR && \
     rm -f test_exec_learning_simple.m && \
     gcc -o test_exec_learning_simple test_exec_learning_simple.c -lm -std=c11 -Wall 2>&1 | grep -E 'error|Error' | head -3 && \
     timeout 300 ./test_exec_learning_simple 2>&1 | tail -30"
echo ""

# Test 3: Production Test
if [ -f "melvin_production_test.c" ]; then
    echo "----------------------------------------"
    echo "TEST 3: Production System Test"
    echo "----------------------------------------"
    sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
        "cd $VM_DIR && \
         rm -f melvin_production_test.m && \
         gcc -o melvin_production_test melvin_production_test.c -lm -std=c11 -Wall 2>&1 | grep -E 'error|Error' | head -3 && \
         timeout 300 ./melvin_production_test 2>&1 | tail -40"
    echo ""
fi

# Test 4: Repeatable Circuits
if [ -f "test_repeatable_circuits.c" ]; then
    echo "----------------------------------------"
    echo "TEST 4: Repeatable Circuits"
    echo "----------------------------------------"
    sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
        "cd $VM_DIR && \
         rm -f test_repeatable_circuits.m && \
         gcc -o test_repeatable_circuits test_repeatable_circuits.c -lm -std=c11 -Wall 2>&1 | grep -E 'error|Error' | head -3 && \
         timeout 300 ./test_repeatable_circuits 2>&1 | tail -30"
    echo ""
fi

# Test 5: Learn Addition Simple
if [ -f "test_learn_addition_simple.c" ]; then
    echo "----------------------------------------"
    echo "TEST 5: Learn Addition Simple"
    echo "----------------------------------------"
    sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
        "cd $VM_DIR && \
         rm -f test_learn_addition_simple.m && \
         gcc -o test_learn_addition_simple test_learn_addition_simple.c -lm -std=c11 -Wall 2>&1 | grep -E 'error|Error' | head -3 && \
         timeout 300 ./test_learn_addition_simple 2>&1 | tail -30"
    echo ""
fi

# Test 6: EXEC Usefulness
if [ -f "test_exec_usefulness.c" ]; then
    echo "----------------------------------------"
    echo "TEST 6: EXEC Usefulness"
    echo "----------------------------------------"
    sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
        "cd $VM_DIR && \
         rm -f test_exec_usefulness.m && \
         gcc -o test_exec_usefulness test_exec_usefulness.c -lm -std=c11 -Wall 2>&1 | grep -E 'error|Error' | head -3 && \
         timeout 300 ./test_exec_usefulness 2>&1 | tail -30"
    echo ""
fi

# Test 7: Phase 2 EXEC Loop
if [ -f "test_phase2_exec_loop.c" ]; then
    echo "----------------------------------------"
    echo "TEST 7: Phase 2 EXEC Loop"
    echo "----------------------------------------"
    sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
        "cd $VM_DIR && \
         rm -f test_phase2_exec_loop.m && \
         gcc -o test_phase2_exec_loop test_phase2_exec_loop.c -lm -std=c11 -Wall 2>&1 | grep -E 'error|Error' | head -3 && \
         timeout 300 ./test_phase2_exec_loop 2>&1 | tail -30"
    echo ""
fi

# Test 8: Phase 2 Prediction Task
if [ -f "test_phase2_prediction_task.c" ]; then
    echo "----------------------------------------"
    echo "TEST 8: Phase 2 Prediction Task"
    echo "----------------------------------------"
    sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no "$VM_USER@$VM_HOST" \
        "cd $VM_DIR && \
         rm -f test_phase2_prediction_task.m && \
         gcc -o test_phase2_prediction_task test_phase2_prediction_task.c -lm -std=c11 -Wall 2>&1 | grep -E 'error|Error' | head -3 && \
         timeout 300 ./test_phase2_prediction_task 2>&1 | tail -30"
    echo ""
fi

echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="
echo ""
echo "Key metrics to compare:"
echo "  - Pattern node count (should be lower with efficiency penalty)"
echo "  - EXEC node usage (should be higher if efficient)"
echo "  - Stability scores (efficient nodes should have higher stability)"
echo "  - Blob size (should reflect efficient code usage)"
echo "  - Prediction accuracy (should maintain or improve)"
echo ""

