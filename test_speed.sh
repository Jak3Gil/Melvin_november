#!/bin/bash
# Speed test for melvin_learn_cli

echo "=== Speed Test ==="
echo ""

# Test 1: Without loading (new graph)
echo "Test 1: New graph (no load)"
time echo "abc" | ./melvin_learn_cli 2>&1 | grep -q "graph_output" && echo "✓ Completed" || echo "✗ Failed"
echo ""

# Test 2: With loading large graph
if [ -f "teacher/melvin_5min_test_graph.bin" ]; then
    echo "Test 2: Load large graph and process"
    time echo "test" | ./melvin_learn_cli --load teacher/melvin_5min_test_graph.bin --save /dev/null 2>&1 | grep -q "graph_output" && echo "✓ Completed" || echo "✗ Failed"
    echo ""
    
    # Test 3: Multiple runs to see consistency
    echo "Test 3: Multiple runs (5x)"
    for i in {1..5}; do
        echo -n "Run $i: "
        (time echo "test$i" | ./melvin_learn_cli --load teacher/melvin_5min_test_graph.bin --save /dev/null > /dev/null 2>&1) 2>&1 | grep real | awk '{print $2}'
    done
else
    echo "Test 2: Skipped (graph file not found)"
fi

echo ""
echo "=== Test Complete ==="

