#!/bin/bash

echo "Testing Melvin's Foundational Beliefs System"
echo "============================================="
echo ""

echo "Starting Melvin with enhanced foundational beliefs..."
echo ""

# Test the foundational beliefs by running Melvin interactively
{
    echo "trace on"
    sleep 1
    echo "hello"
    sleep 2
    echo "how can you help me"
    sleep 2
    echo "what is kindness"
    sleep 2
    echo "tell me about learning"
    sleep 2
    echo "how are you"
    sleep 2
    echo "i am good"
    sleep 2
    echo "how are you"
    sleep 2
    echo "quit"
} | ./melvin_foundational_beliefs

echo ""
echo "Foundational beliefs test completed!"
