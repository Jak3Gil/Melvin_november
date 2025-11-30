#!/bin/bash

echo "Testing Melvin's Ollama Teacher Integration"
echo "=========================================="
echo ""

echo "Starting Melvin with Ollama Teacher Integration..."
echo ""

# Test the Ollama teacher integration
{
    echo "teacher"
    sleep 1
    echo "what is kindness"
    sleep 2
    echo "how can you help me"
    sleep 2
    echo "tell me about learning"
    sleep 2
    echo "teacher status"
    sleep 2
    echo "quit"
} | ./melvin_ollama_teacher

echo ""
echo "Ollama Teacher Integration test completed!"
