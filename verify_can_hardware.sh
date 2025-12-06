#!/bin/bash
# verify_can_hardware.sh - Verify CAN0 hardware is working

echo "╔═══════════════════════════════════════════╗"
echo "║  CAN0 Hardware Verification               ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

if [ "$EUID" -ne 0 ]; then 
    sudo "$0" "$@"
    exit $?
fi

echo "1️⃣  CAN Interface Status..."
echo "═══════════════════════════════════════════"
ip -details link show can0
echo ""

echo "2️⃣  Enabling CAN with loopback for self-test..."
echo "═══════════════════════════════════════════"
ip link set can0 down
ip link set can0 type can bitrate 500000 restart-ms 100
# Enable loopback mode for testing
ip link set can0 type can loopback on
ip link set can0 up

echo "✅ CAN0 configured with loopback"
echo ""

echo "3️⃣  Self-Test (Loopback)..."
echo "═══════════════════════════════════════════"
echo "Sending test frame with loopback enabled..."

timeout 2 candump can0 &
DUMP_PID=$!

sleep 0.5

cansend can0 123#DEADBEEF

sleep 1
kill $DUMP_PID 2>/dev/null

echo ""
echo "If you saw '123  [4]  DE AD BE EF' above, loopback works!"
echo ""

echo "4️⃣  Disabling loopback, testing with motors..."
echo "═══════════════════════════════════════════"
ip link set can0 down
ip link set can0 type can bitrate 500000 restart-ms 100 loopback off
ip link set can0 up

echo "Starting live monitor..."
timeout 10 candump can0 &
DUMP_PID=$!

sleep 1

echo ""
echo "Sending to motors 12 and 14..."
echo "  Motor 12 (0x00C): Enable command..."
cansend can0 00C#A100000000000000
sleep 0.3

echo "  Motor 14 (0x00E): Enable command..."
cansend can0 00E#A100000000000000
sleep 0.3

echo "  Motor 12: Read state..."
cansend can0 00C#9200000000000000
sleep 0.3

echo "  Motor 14: Read state..."
cansend can0 00E#9200000000000000

echo ""
echo "Waiting for motor responses..."
sleep 5

kill $DUMP_PID 2>/dev/null

echo ""
echo "5️⃣  CAN Statistics..."
echo "═══════════════════════════════════════════"
ip -statistics link show can0

echo ""
echo "6️⃣  Analysis..."
echo "═══════════════════════════════════════════"
echo ""
echo "Check TX/RX counters above:"
echo "  • If TX > 0: Commands were sent ✓"
echo "  • If RX > 0: Responses received ✓"
echo "  • If errors > 0: Communication problems ✗"
echo ""
echo "If no RX packets:"
echo "  ☐ Check transceiver wiring (TX, RX, VCC, GND)"
echo "  ☐ Verify CAN-H and CAN-L connections to motors"
echo "  ☐ Confirm 120Ω termination (measure 60Ω)"
echo "  ☐ Check motor power (LEDs blinking?)"
echo ""

