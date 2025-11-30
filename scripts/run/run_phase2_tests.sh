#!/bin/bash

# Phase 2 Test Runner
# Runs all Phase 2 tests in sequence

set -e

echo "=========================================="
echo "PHASE 2 TEST SUITE"
echo "=========================================="
echo ""
echo "Running Phase 2 tests to verify:"
echo "  A. EXEC Loop (pattern→EXEC→code-write→new EXEC)"
echo "  B. Parameter Adaptation (EXEC→param nodes→physics)"
echo "  C. Prediction Task (next-byte prediction with reward)"
echo ""

# Compile all tests
echo "Compiling tests..."
gcc -o test_phase2_exec_loop test_phase2_exec_loop.c -lm -std=c11 -Wall
gcc -o test_phase2_param_adaptation test_phase2_param_adaptation.c -lm -std=c11 -Wall
gcc -o test_phase2_prediction_task test_phase2_prediction_task.c -lm -std=c11 -Wall

echo "✓ All tests compiled"
echo ""

# Run tests
echo "----------------------------------------"
echo "Running: EXEC Loop Test"
echo "----------------------------------------"
./test_phase2_exec_loop
EXEC_LOOP_RESULT=$?
echo ""

echo "----------------------------------------"
echo "Running: Parameter Adaptation Test"
echo "----------------------------------------"
./test_phase2_param_adaptation
PARAM_RESULT=$?
echo ""

echo "----------------------------------------"
echo "Running: Prediction Task Test"
echo "----------------------------------------"
./test_phase2_prediction_task
PREDICTION_RESULT=$?
echo ""

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""

TOTAL_TESTS=3
PASSED=0

if [ $EXEC_LOOP_RESULT -eq 0 ]; then
    echo "✅ EXEC Loop Test: PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ EXEC Loop Test: FAILED"
fi

if [ $PARAM_RESULT -eq 0 ]; then
    echo "✅ Parameter Adaptation Test: PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ Parameter Adaptation Test: FAILED"
fi

if [ $PREDICTION_RESULT -eq 0 ]; then
    echo "✅ Prediction Task Test: PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ Prediction Task Test: FAILED"
fi

echo ""
echo "Total: $PASSED/$TOTAL_TESTS tests passed"

if [ $PASSED -eq $TOTAL_TESTS ]; then
    echo ""
    echo "🎉 ALL PHASE 2 TESTS PASSED!"
    echo "Melvin is ready for Phase 2 deployment."
    exit 0
else
    echo ""
    echo "⚠ SOME TESTS FAILED"
    echo "Review individual test output for details."
    exit 1
fi

