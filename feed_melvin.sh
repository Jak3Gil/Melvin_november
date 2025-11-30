#!/bin/bash
# feed_melvin.sh - Feed files to Melvin brain from Mac
# Usage: ./feed_melvin.sh <file_to_feed> [port_node] [energy]

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
BRAIN_PATH="/mnt/melvin_ssd/melvin/brain.m"
PORT_NODE=${2:-0}
ENERGY=${3:-0.1}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file_to_feed> [port_node] [energy]"
    echo "  port_node: Node ID to feed through (default: 0)"
    echo "  energy: Activation energy per byte (default: 0.1)"
    echo ""
    echo "Examples:"
    echo "  $0 hello.c              # Feed C file"
    echo "  $0 data.txt 0 0.2       # Feed text with higher energy"
    echo "  $0 new_code.c 256       # Feed to specific port node"
    exit 1
fi

FILE_TO_FEED="$1"

if [ ! -f "$FILE_TO_FEED" ]; then
    echo "Error: File not found: $FILE_TO_FEED"
    exit 1
fi

echo "Feeding $FILE_TO_FEED to Melvin brain on Jetson..."
echo "Port node: $PORT_NODE, Energy: $ENERGY"
echo ""

# Transfer file to Jetson
TEMP_FILE="/tmp/$(basename $FILE_TO_FEED)"
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no "$FILE_TO_FEED" "$JETSON_USER@$JETSON_IP:$TEMP_FILE" || {
    echo "Error: Failed to transfer file to Jetson"
    exit 1
}

# Feed file to Melvin
sshpass -p "$JETSON_PASS" ssh "$JETSON_USER@$JETSON_IP" \
    "cd /mnt/melvin_ssd/melvin && ./melvin_feed_file $BRAIN_PATH $TEMP_FILE $PORT_NODE $ENERGY && rm -f $TEMP_FILE"

echo ""
echo "Done! File fed to Melvin brain."

