#!/bin/bash
#
# detect_motors_can.sh - Detect motors on USB-to-CAN bus
#
# Scans CAN bus and listens for motor responses
#

set -e

CAN_INTERFACE="slcan0"

echo "╔═══════════════════════════════════════════╗"
echo "║  Motor Detection on USB-to-CAN Bus        ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

# Check CAN interface
if ! ip link show $CAN_INTERFACE 2>/dev/null | grep -q "UP"; then
    echo "⚠️  $CAN_INTERFACE is not UP"
    echo "   Setting up USB-to-CAN..."
    
    SERIAL_DEVICE=$(ls /dev/ttyUSB* 2>/dev/null | head -1)
    if [ -z "$SERIAL_DEVICE" ]; then
        echo "❌ No USB serial device found"
        exit 1
    fi
    
    pkill slcand 2>/dev/null || true
    sleep 1
    # Start slcand with 921600 baud serial rate
    slcand -o -c -S 921600 -s6 "$SERIAL_DEVICE" $CAN_INTERFACE
    sleep 2
    ip link set $CAN_INTERFACE up
fi

echo "✅ USB-to-CAN Interface: $CAN_INTERFACE"
echo ""

# Show USB device info
echo "📡 USB-to-CAN Adapter:"
lsusb | grep -i "ch340\|serial" | sed 's/^/   /'
echo "   Device: $(ls /dev/ttyUSB* 2>/dev/null | head -1)"
echo ""

# Monitor CAN traffic in background
echo "🔍 Monitoring CAN bus for 10 seconds..."
echo "   (Watch for motor responses)"
echo ""

# Start candump in background and capture output
timeout 10 candump $CAN_INTERFACE > /tmp/can_monitor.log 2>&1 &
CANDUMP_PID=$!

sleep 1

# Send enable commands to all possible motor IDs
echo "📤 Sending enable commands to motors 0-13..."
echo ""

MOTORS_FOUND=0

for i in {0..13}; do
    # Motor CAN IDs: 0x01 to 0x0E (motors 0-13)
    can_id=$(printf '%02X' $((0x01 + $i)))
    
    # Send enable command (Robstride protocol: 0xA1 = enable)
    cansend $CAN_INTERFACE ${can_id}#A100000000000000 > /dev/null 2>&1
    
    # Small delay
    usleep 100000
done

echo ""
echo "⏳ Waiting for responses..."
sleep 3

# Stop candump
kill $CANDUMP_PID 2>/dev/null || true
wait $CANDUMP_PID 2>/dev/null || true

# Analyze captured traffic
echo ""
echo "═══════════════════════════════════════════"
echo "📊 CAN Traffic Analysis:"
echo "═══════════════════════════════════════════"
echo ""

if [ -s /tmp/can_monitor.log ]; then
    echo "✅ CAN traffic detected:"
    echo ""
    
    # Show all frames
    cat /tmp/can_monitor.log | head -20
    
    # Count unique CAN IDs
    UNIQUE_IDS=$(cat /tmp/can_monitor.log | awk '{print $3}' | sort -u | wc -l)
    echo ""
    echo "📈 Found $UNIQUE_IDS unique CAN IDs"
    
    # Map CAN IDs to motor numbers
    echo ""
    echo "🔢 Motor Mapping:"
    cat /tmp/can_monitor.log | awk '{print $3}' | sort -u | while read can_id; do
        # Remove leading zeros and convert hex to decimal
        motor_num=$((0x$can_id - 1))
        if [ $motor_num -ge 0 ] && [ $motor_num -le 13 ]; then
            echo "   Motor $motor_num: CAN ID $can_id"
            MOTORS_FOUND=$((MOTORS_FOUND + 1))
        fi
    done
    
    echo ""
    echo "✅ Found $MOTORS_FOUND motors responding!"
    
else
    echo "⚠️  No CAN traffic detected"
    echo ""
    echo "Possible reasons:"
    echo "   1. Motors are not powered"
    echo "   2. Motors not connected to CAN bus"
    echo "   3. Wrong CAN bitrate"
    echo "   4. CAN bus wiring issue"
    echo ""
    echo "Try:"
    echo "   - Check motor power LEDs"
    echo "   - Verify CAN-H and CAN-L connections"
    echo "   - Check CAN bus termination (120Ω resistors)"
fi

# Test specific motors 12 and 14
echo ""
echo "═══════════════════════════════════════════"
echo "🎯 Testing Motors 12 and 14 specifically:"
echo "═══════════════════════════════════════════"
echo ""

for motor in 12 14; do
    can_id=$(printf '%02X' $motor)
    echo -n "Motor $motor (0x$can_id): "
    
    # Send enable and listen
    timeout 1 candump $CAN_INTERFACE 2>/dev/null > /tmp/motor_${motor}_test.log &
    CANDUMP_PID=$!
    sleep 0.2
    cansend $CAN_INTERFACE ${can_id}#A100000000000000 > /dev/null 2>&1
    sleep 0.5
    kill $CANDUMP_PID 2>/dev/null || true
    
    if [ -s /tmp/motor_${motor}_test.log ]; then
        echo "✅ RESPONDING"
        cat /tmp/motor_${motor}_test.log | head -3 | sed 's/^/      /'
    else
        echo "❌ No response"
    fi
done

echo ""
echo "✅ Detection complete!"
echo ""

