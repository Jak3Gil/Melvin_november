#!/bin/bash

echo "Testing Melvin's Cross-Thread Generalization & Wants-Based Traversal"
echo "=================================================================="
echo ""

echo "Starting Melvin with enhanced wants and generalization systems..."
echo ""

# Test the cross-thread generalization and wants-based traversal
{
    echo "trace on"
    sleep 1
    echo "how are you"
    sleep 2
    echo "i am good"
    sleep 2
    echo "how are you"
    sleep 2
    echo "i'm fine"
    sleep 2
    echo "how are you"
    sleep 2
    echo "i am doing well"
    sleep 2
    echo "how are you"
    sleep 2
    echo "how are you"
    sleep 2
    echo "quit"
} | ./melvin_wants_generalization

echo ""
echo "Cross-thread generalization and wants-based traversal test completed!"
