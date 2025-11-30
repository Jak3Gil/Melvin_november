#!/bin/bash
# Run all tests on melvin.m with instincts injected

set -e

MELVIN_FILE="melvin.m"

echo "=== STEP 1: Create fresh melvin.m with instincts ==="
./test_with_instincts "$MELVIN_FILE"

echo ""
echo "=== STEP 2: Run learning kernel test ==="
./test_learning_kernel

echo ""
echo "=== STEP 3: Run evolution diagnostic test ==="
rm -f evolution_compounding.m evolution_multistep.m evolution_learning.m
./test_evolution_diagnostic

echo ""
echo "=== STEP 4: Run universal laws tests ==="
rm -f test_universal_laws.m
./test_universal_laws

echo ""
echo "=== ALL TESTS COMPLETE ==="
