#!/bin/bash
# Melvin Graph Capabilities Test
# Tests what the graph can learn, remember, and generate

set -e

echo "=========================================="
echo "MELVIN GRAPH CAPABILITIES TEST"
echo "=========================================="
echo ""

BINARY="./melvin_learn_cli"
SNAPSHOT="/tmp/melvin_capabilities_test.snap"

# Clean up old snapshot
rm -f "$SNAPSHOT"

echo "Test 1: Basic Pattern Learning"
echo "-----------------------------"
echo "Input: 'abc'"
echo "abc" | $BINARY --load /dev/null --save "$SNAPSHOT" 2>&1 | grep -E "(num_patterns|compression_ratio|reconstruction_error)" | head -3
echo ""

echo "Test 2: Sequence Repetition"
echo "----------------------------"
echo "Input: 'abcabc'"
echo "abcabc" | $BINARY --load "$SNAPSHOT" --save "$SNAPSHOT" 2>&1 | grep -E "(num_patterns|compression_ratio|reconstruction_error)" | head -3
echo ""

echo "Test 3: Number Sequences"
echo "------------------------"
echo "Input: '1 2 3 4 5'"
echo "1 2 3 4 5" | $BINARY --load "$SNAPSHOT" --save "$SNAPSHOT" 2>&1 | grep -E "(num_patterns|compression_ratio|reconstruction_error)" | head -3
echo ""

echo "Test 4: Alphabet Pattern"
echo "------------------------"
echo "Input: 'a b c d e'"
echo "a b c d e" | $BINARY --load "$SNAPSHOT" --save "$SNAPSHOT" 2>&1 | grep -E "(num_patterns|compression_ratio|reconstruction_error)" | head -3
echo ""

echo "Test 5: Mixed Patterns"
echo "----------------------"
echo "Input: 'hello world'"
echo "hello world" | $BINARY --load "$SNAPSHOT" --save "$SNAPSHOT" 2>&1 | grep -E "(num_patterns|compression_ratio|reconstruction_error)" | head -3
echo ""

echo "Test 6: Graph Output Generation"
echo "-------------------------------"
echo "Input: 'test'"
echo "test" | $BINARY --load "$SNAPSHOT" --save "$SNAPSHOT" 2>&1 | grep -E "graph_output" | head -1
echo ""

echo "Test 7: Graph Statistics"
echo "-------------------------"
echo "Checking final graph state..."
echo "test" | $BINARY --load "$SNAPSHOT" --save /dev/null 2>&1 | grep -E "(num_patterns|nodes|edges)" | head -3
echo ""

echo "Test 8: Snapshot Load Speed"
echo "---------------------------"
echo "Testing snapshot load time..."
time echo "test" | $BINARY --load "$SNAPSHOT" --save /dev/null 2>&1 | grep "graph_load_auto" | head -1
echo ""

echo "=========================================="
echo "CAPABILITIES TEST COMPLETE"
echo "=========================================="

