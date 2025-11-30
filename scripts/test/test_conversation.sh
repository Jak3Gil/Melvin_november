#!/bin/bash

echo "Testing Melvin's Enhanced Conversation Mode with Ollama Integration"
echo "=================================================================="
echo ""

# Test the conversation mode
echo "Starting conversation mode test..."
echo "conversation" | timeout 30s ./melvin_enhanced

echo ""
echo "Test completed!"
