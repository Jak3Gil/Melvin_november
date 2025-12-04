#!/bin/bash
# Scan ALL CAN IDs to find motors

echo "╔═══════════════════════════════════════════╗"
echo "║  CAN Motor Scanner                        ║"
echo "║  Scanning IDs 1-127 for responses        ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Start candump in background
timeout 30 candump -c slcan0 > /tmp/can_scan.log 2>&1 &
DUMP_PID=$!

sleep 1

echo "Sending Enter Control to IDs 1-30..."
echo "(This will take ~30 seconds)"
echo ""

for id in {1..30}; do
    hex_id=$(printf "%03X" $id)
    printf "  ID %2d (0x%s)... " $id $hex_id
    
    # Send enter control mode command
    sudo cansend slcan0 ${hex_id}#FFFFFFFFFFFFFFFC 2>/dev/null
    sleep 0.1
    
    # Check if we got a response
    if grep -q "$hex_id" /tmp/can_scan.log 2>/dev/null; then
        echo "✅ RESPONDED!"
    else
        echo "no response"
    fi
done

echo ""
echo "Stopping scanner..."
kill $DUMP_PID 2>/dev/null

echo ""
echo "═══════════════════════════════════════════"
echo "Results:"
echo "═══════════════════════════════════════════"

if [ -s /tmp/can_scan.log ]; then
    echo ""
    cat /tmp/can_scan.log
    echo ""
    
    UNIQUE_IDS=$(cat /tmp/can_scan.log | awk '{print $2}' | sort -u)
    
    if [ -n "$UNIQUE_IDS" ]; then
        echo "✅ Found motors at these IDs:"
        echo "$UNIQUE_IDS"
    else
        echo "❌ No motor responses detected"
    fi
else
    echo "❌ No CAN traffic at all"
fi

echo ""

