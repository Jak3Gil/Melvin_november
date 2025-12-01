#!/bin/bash
# Quick test to prove .m can learn both pattern learning and EXEC computation

BRAIN_FILE="${1:-test/test_brain.m}"

echo "=========================================="
echo "TEST: Brain Learning Both Methods"
echo "=========================================="
echo "Brain file: $BRAIN_FILE"
echo ""

# Check if brain exists
if [ ! -f "$BRAIN_FILE" ]; then
    echo "Creating new brain..."
    ./scripts/create_new_brain.sh "$BRAIN_FILE"
    echo ""
fi

echo "Step 1: Seeding pattern learning examples..."
echo "  Feeding: 2+3=5, 4+1=5, 10+20=30, etc."
melvin_seed_knowledge "$BRAIN_FILE" corpus/math/pattern_examples.txt 0.4
echo ""

echo "Step 2: Creating EXEC computation nodes..."
melvin_seed_arithmetic_exec "$BRAIN_FILE" 1.0
echo ""

echo "Step 3: Connecting patterns to EXEC..."
melvin_seed_patterns "$BRAIN_FILE" corpus/math/exec_operations.txt 0.6
echo ""

echo "Step 4: Running learning test..."
if [ -f "./test/test_learn_both_methods" ]; then
    # Update test to use our brain file
    TEST_BRAIN="test_learn_both.m"
    cp "$BRAIN_FILE" "$TEST_BRAIN"
    ./test/test_learn_both_methods
    rm -f "$TEST_BRAIN"
else
    echo "  ⚠ Test executable not found - compile with: make test/test_learn_both_methods"
fi

echo ""
echo "=========================================="
echo "Brain file ready: $BRAIN_FILE"
echo "=========================================="

