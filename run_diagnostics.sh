#!/bin/bash

# Melvin Diagnostics Runner
# Runs all three diagnostic experiments to verify physics

set -e

echo "=========================================="
echo "MELVIN DIAGNOSTICS SUITE"
echo "=========================================="
echo ""
echo "This will run three diagnostic experiments:"
echo "  A: Single-node sanity check"
echo "  B: Pattern learning (structured vs random)"
echo "  C: Control loop concept check"
echo ""

# Compile diagnostics
echo "Compiling diagnostic tools..."
gcc -o diag_experiment_a_single_node diag_experiment_a_single_node.c -lm -std=c11 -Wall -O2
gcc -o diag_experiment_b_pattern_learning diag_experiment_b_pattern_learning.c -lm -std=c11 -Wall -O2
gcc -o diag_experiment_c_control_loop diag_experiment_c_control_loop.c -lm -std=c11 -Wall -O2

echo "✓ Compilation complete"
echo ""

# Create results directory
mkdir -p diagnostics_results

# Run Experiment A
echo "=========================================="
echo "EXPERIMENT A: Single-Node Sanity"
echo "=========================================="
./diag_experiment_a_single_node
EXPERIMENT_A_RESULT=$?
echo ""

# Run Experiment B
echo "=========================================="
echo "EXPERIMENT B: Pattern Learning"
echo "=========================================="
./diag_experiment_b_pattern_learning
EXPERIMENT_B_RESULT=$?
echo ""

# Run Experiment C
echo "=========================================="
echo "EXPERIMENT C: Control Loop Check"
echo "=========================================="
./diag_experiment_c_control_loop
EXPERIMENT_C_RESULT=$?
echo ""

# Summary
echo "=========================================="
echo "DIAGNOSTICS SUMMARY"
echo "=========================================="
echo ""

TOTAL_EXPERIMENTS=3
PASSED=0

if [ $EXPERIMENT_A_RESULT -eq 0 ]; then
    echo "✓ Experiment A: PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ Experiment A: FAILED"
fi

if [ $EXPERIMENT_B_RESULT -eq 0 ]; then
    echo "✓ Experiment B: PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ Experiment B: FAILED"
fi

if [ $EXPERIMENT_C_RESULT -eq 0 ]; then
    echo "✓ Experiment C: PASSED"
    PASSED=$((PASSED + 1))
else
    echo "❌ Experiment C: FAILED"
fi

echo ""
echo "Results: $PASSED/$TOTAL_EXPERIMENTS experiments passed"
echo ""
echo "Diagnostic logs saved in:"
echo "  - diag_a_results/"
echo "  - diag_b_structured_results/"
echo "  - diag_b_random_results/"
echo "  - diag_c_no_learning_results/"
echo "  - diag_c_with_learning_results/"
echo ""

if [ $PASSED -eq $TOTAL_EXPERIMENTS ]; then
    echo "=========================================="
    echo "ALL DIAGNOSTICS PASSED"
    echo "=========================================="
    echo ""
    echo "Physics verification complete."
    echo "Ready for instincts.m training."
    exit 0
else
    echo "=========================================="
    echo "SOME DIAGNOSTICS FAILED"
    echo "=========================================="
    echo ""
    echo "Review diagnostic logs to identify issues."
    echo "Check CSV files for detailed time-series data."
    exit 1
fi

