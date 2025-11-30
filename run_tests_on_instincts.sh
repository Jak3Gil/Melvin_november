#!/bin/bash
# Run all tests, using melvin_with_instincts.m if available

set -e

echo "=== COMPREHENSIVE TEST SUITE ==="
echo ""

# Check if melvin_with_instincts.m exists
if [ -f melvin_with_instincts.m ]; then
    echo "✓ Found melvin_with_instincts.m"
    ls -lh melvin_with_instincts.m
    echo ""
else
    echo "⚠ melvin_with_instincts.m not found, tests will create their own files"
    echo ""
fi

# Test 1: Learning Kernel
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Learning Kernel"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f test_learning_kernel ]; then
    ./test_learning_kernel 2>&1
    echo ""
else
    echo "⚠ test_learning_kernel not found"
    echo ""
fi

# Test 2: Evolution Diagnostic
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Evolution Diagnostic (Multi-run Compounding)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f test_evolution_diagnostic ]; then
    rm -f evolution_compounding.m evolution_multistep.m evolution_learning.m
    echo "Phase 1:"
    ./test_evolution_diagnostic --phase=1 2>&1 | head -60
    echo ""
    echo "Phase 2:"
    ./test_evolution_diagnostic --phase=2 2>&1 | head -60
    echo ""
else
    echo "⚠ test_evolution_diagnostic not found"
    echo ""
fi

# Test 3: Universal Laws
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Universal Laws (Comprehensive)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f test_universal_laws ]; then
    rm -f test_*.m
    echo "Running comprehensive universal laws test..."
    ./test_universal_laws 2>&1 | tail -100
    echo ""
else
    echo "⚠ test_universal_laws not found"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ALL TESTS COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
