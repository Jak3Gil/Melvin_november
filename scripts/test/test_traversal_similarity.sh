#!/bin/bash

echo "Testing Melvin's Traversal-Based Similarity System"
echo "================================================="
echo ""

echo "Starting Melvin with enhanced traversal-based similarity..."
echo ""

# Test the traversal-based similarity by running Melvin interactively
{
    echo "trace on"
    sleep 1
    echo "hello"
    sleep 2
    echo "whats up"
    sleep 2
    echo "hi there"
    sleep 2
    echo "how are you"
    sleep 2
    echo "i am good"
    sleep 2
    echo "i'm fine"
    sleep 2
    echo "whats up"
    sleep 2
    echo "hello"
    sleep 2
    echo "quit"
} | ./melvin_traversal_similarity

echo ""
echo "Traversal-based similarity test completed!"
