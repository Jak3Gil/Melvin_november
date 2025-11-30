#!/bin/bash
# Test script to verify memory layout fixes

echo "=== Testing Memory Layout Fixes ==="
echo ""

# Run melvin and capture output
echo "Starting melvin test..."
./melvin 2>&1 | tee /tmp/melvin_test.log &
MELVIN_PID=$!

# Wait a bit for initialization
sleep 3

# Check for layout verification
echo ""
echo "=== Layout Check Results ==="
grep -A 5 "LAYOUT CHECK" /tmp/melvin_test.log | head -10

# Check for self-test results
echo ""
echo "=== Self-Test Results ==="
grep -E "(selftest|PASS|FAIL)" /tmp/melvin_test.log | head -10

# Check for errors
echo ""
echo "=== Error Check ==="
grep -iE "(ERROR|CRITICAL|corruption|0x3c23d70a3f000000)" /tmp/melvin_test.log | head -20

# Check edge creation
echo ""
echo "=== Edge Creation Status ==="
grep -E "(CREATING edge|num_edges)" /tmp/melvin_test.log | tail -5

# Kill melvin
kill $MELVIN_PID 2>/dev/null
wait $MELVIN_PID 2>/dev/null

echo ""
echo "=== Test Complete ==="

