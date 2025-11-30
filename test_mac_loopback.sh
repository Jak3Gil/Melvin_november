#!/bin/bash
# test_mac_loopback.sh - Continuous audio loopback test for Mac
# Records from mic and plays to speaker in real-time

echo "=========================================="
echo "USB Audio Loopback Test - Mac"
echo "=========================================="
echo ""
echo "This will pipe microphone → speaker in real-time"
echo "Speak into the microphone - you should hear it immediately!"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start loopback
sox -d -r 16000 -c 1 -b 16 -t wav - | sox -t wav - -d -r 16000 -c 1 -b 16

