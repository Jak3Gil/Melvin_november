#!/bin/bash

echo "Testing Melvin's Sequential Linking System"
echo "========================================="
echo ""

echo "Starting Melvin with sequential linking..."
echo ""

# Test the sequential linking by running Melvin interactively
{
    echo "trace on"
    sleep 1
    echo "hello"
    sleep 2
    echo "how are you"
    sleep 2
    echo "you could say i am good"
    sleep 2
    echo "how are you"
    sleep 2
    echo "quit"
} | ./melvin_sequential

echo ""
echo "Sequential linking test completed!"
