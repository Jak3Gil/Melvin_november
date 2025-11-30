#!/bin/bash

# Run all theoretical capability tests
# This script tests what's theoretically possible vs what actually works

set -e

echo "=========================================="
echo "THEORETICAL CAPABILITY TESTS"
echo "=========================================="
echo ""
echo "Testing theoretical capabilities from MASTER_ARCHITECTURE.md"
echo ""

# Compile all tests
echo "Compiling tests..."
gcc -o test_self_modify test_self_modify.c -lm -std=c11 -Wall
gcc -o test_code_evolution test_code_evolution.c -lm -std=c11 -Wall
gcc -o test_auto_exec test_auto_exec.c -lm -std=c11 -Wall
gcc -o test_meta_learning test_meta_learning.c -lm -std=c11 -Wall
gcc -o test_emergent_algo test_emergent_algo.c -lm -std=c11 -Wall

echo "✓ All tests compiled"
echo ""

# Run tests
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

run_test() {
    local test_name=$1
    local test_binary=$2
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "----------------------------------------"
    
    if ./$test_binary; then
        echo ""
        echo "✅ $test_name: PASSED"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo ""
        echo "❌ $test_name: FAILED"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
}

# Run each test
run_test "Self-Modifying Code" "test_self_modify"
run_test "Code Evolution" "test_code_evolution"
run_test "Automatic EXEC Creation" "test_auto_exec"
run_test "Meta-Learning" "test_meta_learning"
run_test "Emergent Algorithm Formation" "test_emergent_algo"

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Total tests: $TESTS_TOTAL"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    echo ""
    echo "All theoretical capabilities are working!"
    exit 0
else
    echo "⚠ SOME TESTS FAILED"
    echo ""
    echo "Some theoretical capabilities may not be fully implemented."
    echo "Check individual test output for details."
    exit 1
fi

