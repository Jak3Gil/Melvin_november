#!/bin/bash
#
# diagnose_can.sh - Diagnose CAN Bus Issues
#

echo "╔═══════════════════════════════════════════╗"
echo "║     CAN Bus Diagnostic Tool               ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

echo "1️⃣  Checking CAN interface..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! ip link show can0 &>/dev/null; then
    echo "❌ can0 not found"
    echo ""
    echo "Available interfaces:"
    ip link show | grep -E "^[0-9]+" | cut -d: -f2
    echo ""
    echo "Try setting up slcand for USB-CAN adapter:"
    echo "  sudo slcand -o -c -s6 /dev/ttyUSB0 can0"
    exit 1
fi

echo "✅ can0 exists"
echo ""

echo "2️⃣  CAN Interface Details..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ip -details link show can0
echo ""

echo "3️⃣  CAN Statistics..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ip -statistics link show can0
echo ""

echo "4️⃣  Testing CAN Bus (5 second listen)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting candump..."
timeout 5 candump can0 2>&1 | tee /tmp/can_traffic.log &
CANDUMP_PID=$!

sleep 1

echo ""
echo "Sending test frames to motors 13 and 14..."

# Try sending with different formats
echo "  Testing Motor 13 (0x0D) - Enable command..."
cansend can0 00D#A100000000000000 2>/dev/null && echo "    ✓ Sent" || echo "    ✗ Failed: $?"

sleep 0.5

echo "  Testing Motor 14 (0x0E) - Enable command..."
cansend can0 00E#A100000000000000 2>/dev/null && echo "    ✓ Sent" || echo "    ✗ Failed: $?"

sleep 0.5

echo "  Testing Motor 13 - Read State command..."
cansend can0 00D#9200000000000000 2>/dev/null && echo "    ✓ Sent" || echo "    ✗ Failed: $?"

sleep 2

wait $CANDUMP_PID 2>/dev/null

echo ""
echo "5️⃣  CAN Traffic Analysis..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -s /tmp/can_traffic.log ]; then
    echo "✅ CAN traffic detected:"
    cat /tmp/can_traffic.log
    echo ""
    FRAME_COUNT=$(wc -l < /tmp/can_traffic.log)
    echo "Total frames: $FRAME_COUNT"
else
    echo "❌ NO CAN traffic detected"
    echo ""
    echo "This means:"
    echo "  • Motors may not be powered on"
    echo "  • Motors not connected to CAN bus"
    echo "  • Wrong CAN IDs (try scanning)"
    echo "  • Missing bus termination (120Ω resistors)"
fi

echo ""
echo "6️⃣  Error State Check..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STATE=$(ip -details link show can0 | grep "state" | awk '{print $2}')
echo "CAN State: $STATE"

if [ "$STATE" != "ERROR-ACTIVE" ]; then
    echo "⚠️  CAN bus is in error state!"
    echo "Resetting..."
    ip link set can0 down
    sleep 1
    ip link set can0 type can bitrate 500000 restart-ms 100
    ip link set can0 up
    echo "✅ Reset complete"
fi

echo ""
echo "7️⃣  USB-CAN Adapter Check..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if lsusb | grep -i "can\|ch341\|cp210\|ftdi\|Peak"; then
    echo "✅ USB-CAN adapter detected:"
    lsusb | grep -i "can\|ch341\|cp210\|ftdi\|Peak"
else
    echo "⚠️  No obvious USB-CAN adapter found"
    echo ""
    echo "All USB devices:"
    lsusb
fi

echo ""
echo "8️⃣  Recommendations..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -s /tmp/can_traffic.log ]; then
    echo ""
    echo "Since no CAN traffic was detected, check:"
    echo ""
    echo "  1. Motor Power:"
    echo "     ☐ Are motors powered on?"
    echo "     ☐ Check power LED on motors"
    echo "     ☐ Verify voltage (usually 24V or 48V)"
    echo ""
    echo "  2. Physical Connection:"
    echo "     ☐ USB-CAN adapter plugged in?"
    echo "     ☐ CAN-H and CAN-L wires connected?"
    echo "     ☐ Check for loose connections"
    echo ""
    echo "  3. Bus Termination:"
    echo "     ☐ 120Ω resistor at START of bus"
    echo "     ☐ 120Ω resistor at END of bus"
    echo "     ☐ Measure: should see ~60Ω between CAN-H and CAN-L"
    echo ""
    echo "  4. Motor Configuration:"
    echo "     ☐ Motors set to CAN IDs 13 (0x0D) and 14 (0x0E)?"
    echo "     ☐ Motors configured for 500kbps CAN bitrate?"
    echo "     ☐ Check motor documentation"
    echo ""
    echo "  5. Try Manual Scan:"
    echo "     # Scan all possible CAN IDs"
    echo "     for id in {1..127}; do"
    echo "       printf \"Trying 0x%02X...\\n\" \$id"
    echo "       cansend can0 \$(printf \"%03X\" \$id)#9200000000000000"
    echo "       sleep 0.1"
    echo "     done"
    echo ""
else
    echo "✅ CAN bus appears functional!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: sudo /tmp/test_robstride_motors"
    echo "  2. Watch motors move slowly"
    echo "  3. Document what motors 13 and 14 control"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "Diagnostic complete!"
echo "═══════════════════════════════════════════"

