#!/bin/bash

echo "Testing Melvin's Temporal Priority & Conversation Thread Clustering"
echo "=================================================================="
echo ""

echo "Starting Melvin with enhanced temporal and thread systems..."
echo ""

# Test the temporal priority and thread clustering
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
    echo "what is your name"
    sleep 2
    echo "my name is melvin"
    sleep 2
    echo "what is your name"
    sleep 2
    echo "quit"
} | ./melvin_temporal_threads

echo ""
echo "Temporal priority and thread clustering test completed!"
