#!/bin/bash
# Speed comparison test

echo "=== Speed Test: Current Implementation ==="
echo ""

# Test with different input sizes
for input in "a" "abc" "test" "hello world"; do
    echo "Input: '$input'"
    (time echo "$input" | ./melvin_learn_cli --load teacher/melvin_5min_test_graph.bin --save /dev/null > /dev/null 2>&1) 2>&1 | grep real | awk '{print "  Time: " $2}'
done

echo ""
echo "=== Summary ==="
echo "Current approach: Uses input length for learning/activation (intelligent scaling)"
echo "This scales with input complexity, not arbitrary limits"

