#!/bin/bash
# Check if motors are actually responding

echo "🔍 Checking Motor Feedback"
echo "=========================="
echo ""

# Start monitoring
echo "Starting CAN monitor..."
candump -c -a slcan0 > /tmp/can_check.log 2>&1 &
DUMP_PID=$!

sleep 1

echo "Sending commands to motors..."
echo ""

echo "1. Enable Motor 12 (0x0C)..."
cansend slcan0 00C#A100000000000000
sleep 0.5

echo "2. Read State from Motor 12..."
cansend slcan0 00C#9200000000000000
sleep 0.5

echo "3. Enable Motor 14 (0x0E)..."
cansend slcan0 00E#A100000000000000
sleep 0.5

echo "4. Read State from Motor 14..."
cansend slcan0 00E#9200000000000000
sleep 2

echo ""
echo "Stopping monitor..."
kill $DUMP_PID 2>/dev/null
sleep 0.5

echo ""
echo "═══════════════════════════════════════"
echo "CAN Traffic Analysis"
echo "═══════════════════════════════════════"
echo ""

if [ -s /tmp/can_check.log ]; then
    cat /tmp/can_check.log
    echo ""
    
    TOTAL=$(wc -l < /tmp/can_check.log)
    SENT=$(grep -c "00C\|00E" /tmp/can_check.log || echo 0)
    
    echo "Total frames: $TOTAL"
    echo "Our commands: $SENT"
    echo "Motor responses: $((TOTAL - SENT))"
    echo ""
    
    if [ $((TOTAL - SENT)) -eq 0 ]; then
        echo "❌ NO motor responses detected!"
        echo ""
        echo "This means:"
        echo "  • Motors are not acknowledging commands"
        echo "  • Motors may not be powered"
        echo "  • Wrong CAN IDs"
        echo "  • Motors in error state"
        echo ""
        echo "Suggestions:"
        echo "  1. Check motor power LEDs"
        echo "  2. Verify motor CAN IDs (should be 12=0x0C, 14=0x0E)"
        echo "  3. Check if motors need initialization sequence"
        echo "  4. Verify CAN wiring is correct"
    else
        echo "✅ Motors ARE responding!"
        echo ""
        echo "But you didn't see movement, which means:"
        echo "  • Motors are in wrong control mode"
        echo "  • Motors need different command format"
        echo "  • Position control is disabled"
        echo "  • Motors are mechanically stuck"
    fi
else
    echo "❌ No CAN traffic captured at all"
    echo "Check if slcan0 is up: ip link show slcan0"
fi

