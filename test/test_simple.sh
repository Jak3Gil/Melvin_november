#!/bin/bash
# Simple test script - just feeds bytes to brain and shows results
# No new C files needed - just use melvin_run

BRAIN="test_brain.m"

echo "=== Test 1: Feed 'A' pattern ==="
echo "AAAA" | ./melvin_run "$BRAIN"

echo ""
echo "=== Test 2: Feed 'AB' pattern ==="
echo "ABAB" | ./melvin_run "$BRAIN"

echo ""
echo "=== Test 3: Feed C code ==="
echo "int main() { return 0; }" | ./melvin_run "$BRAIN"

echo ""
echo "Brain file: $BRAIN"
echo "To inspect, use a tool that reads the .m file directly"

