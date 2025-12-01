#!/bin/bash
# Complete math seeding script - teaches both pattern learning and EXEC computation

if [ $# -lt 1 ]; then
    echo "Usage: $0 <melvin.m file>"
    echo "  Seeds complete math foundation: patterns, examples, and EXEC nodes"
    exit 1
fi

BRAIN_FILE="$1"

echo "=========================================="
echo "COMPLETE MATH FOUNDATION SEEDING"
echo "=========================================="
echo "Brain file: $BRAIN_FILE"
echo ""

# Check if brain file exists
if [ ! -f "$BRAIN_FILE" ]; then
    echo "Error: Brain file not found: $BRAIN_FILE"
    exit 1
fi

# Phase 1: Bootstrap patterns
echo "Phase 1: Bootstrap patterns..."
melvin_seed_patterns "$BRAIN_FILE" corpus/basic/patterns.txt 0.6
echo ""

# Phase 2: Math concept patterns
echo "Phase 2: Math concept patterns..."
melvin_seed_patterns "$BRAIN_FILE" corpus/math/arithmetic.txt 0.5
melvin_seed_patterns "$BRAIN_FILE" corpus/math/algebra.txt 0.5
melvin_seed_patterns "$BRAIN_FILE" corpus/math/geometry.txt 0.4
echo ""

# Phase 3: Pattern learning examples
echo "Phase 3: Pattern learning examples..."
melvin_seed_knowledge "$BRAIN_FILE" corpus/math/pattern_examples.txt 0.4
echo ""

# Phase 4: EXEC computation nodes
echo "Phase 4: EXEC computation nodes..."
melvin_seed_arithmetic_exec "$BRAIN_FILE" 1.0
echo ""

# Phase 5: Connect patterns to EXEC
echo "Phase 5: Connect patterns to EXEC nodes..."
melvin_seed_patterns "$BRAIN_FILE" corpus/math/exec_operations.txt 0.6
echo ""

# Phase 6: Teach computation methods
echo "Phase 6: Teach when to use each method..."
melvin_seed_patterns "$BRAIN_FILE" corpus/math/computation.txt 0.5
echo ""

echo "=========================================="
echo "COMPLETE MATH FOUNDATION SEEDED"
echo "=========================================="
echo ""
echo "The system now has:"
echo "  ✓ Pattern learning (from examples)"
echo "  ✓ EXEC computation (direct CPU operations)"
echo "  ✓ Connections between patterns and EXEC"
echo "  ✓ Knowledge of when to use each method"
echo ""
echo "Both methods are ready to use!"

