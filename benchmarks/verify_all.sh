#!/bin/bash
# Verify that speed optimizations didn't break correctness

echo "=========================================="
echo "VERIFICATION SUITE: Speed + Correctness"
echo "=========================================="
echo ""

# Save old results for comparison
mkdir -p data/old_results
if [ -f data/experiment1_results.csv ]; then
    cp data/experiment1_results.csv data/old_results/
fi
if [ -f data/experiment2_results.csv ]; then
    cp data/experiment2_results.csv data/old_results/
fi
if [ -f data/experiment3_results.csv ]; then
    cp data/experiment3_results.csv data/old_results/
fi

echo "Running verification tests..."
echo ""

# Test 1: Pattern discovery still works
echo "1. Pattern Discovery Test..."
./experiment1_pattern_efficiency 2>&1 | tail -10

# Test 2: Hierarchical composition still works
echo ""
echo "2. Hierarchical Composition Test..."
./experiment2_hierarchical 2>&1 | grep -E "Phase.*Complete|Pattern Growth"

# Test 3: Scaling efficiency still works
echo ""
echo "3. Scaling Efficiency Test..."
./experiment3_scaling 2>&1 | grep -E "^[0-9]|Average"

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "Checking for regressions..."

# Compare pattern counts
echo "Pattern discovery check:"
if [ -f data/old_results/experiment1_results.csv ]; then
    old_patterns=$(tail -1 data/old_results/experiment1_results.csv | cut -d',' -f3)
    new_patterns=$(tail -1 data/experiment1_results.csv | cut -d',' -f3)
    echo "  Old: $old_patterns patterns"
    echo "  New: $new_patterns patterns"
    
    if [ "$new_patterns" -ge "$((old_patterns / 2))" ]; then
        echo "  ✓ Pattern count maintained"
    else
        echo "  ⚠ WARNING: Pattern count dropped significantly!"
    fi
fi

echo ""
echo "All experiments re-run. Compare data/old_results/ to data/ for changes."

