#!/bin/bash

echo "Testing Melvin's Enhanced Features"
echo "=================================="
echo ""

echo "1. Testing Ollama status command..."
echo "ollama" | ./melvin_enhanced | head -20

echo ""
echo "2. Testing adaptation stats..."
echo "adaptation" | ./melvin_enhanced | head -20

echo ""
echo "3. Testing analytics..."
echo "analytics" | ./melvin_enhanced | head -20

echo ""
echo "4. Testing trace mode..."
echo "trace on" | ./melvin_enhanced | head -10

echo ""
echo "Feature tests completed!"
