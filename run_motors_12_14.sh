#!/bin/bash
#
# run_motors_12_14.sh - Quick script to test motors 12 and 14
#
# This assumes CAN is already set up. If not, run setup_usb_can_motors.sh first
#

set -e

echo "╔═══════════════════════════════════════════╗"
echo "║  Test Motors 12 and 14                    ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

# Check if USB-to-CAN interface is up (slcan0 or can0)
CAN_INTERFACE="slcan0"
if ! ip link show $CAN_INTERFACE 2>/dev/null | grep -q "UP"; then
    # Try can0 as fallback
    if ip link show can0 2>/dev/null | grep -q "UP"; then
        CAN_INTERFACE="can0"
    else
        echo "⚠️  USB-to-CAN interface not up"
        echo ""
        echo "Setting up USB-to-CAN interface..."
        
        # Try to find USB device
        SERIAL_DEVICE=$(ls /dev/ttyUSB* 2>/dev/null | head -1)
        
        if [ -z "$SERIAL_DEVICE" ]; then
            echo "❌ No USB serial device found"
            echo ""
            echo "Please run setup first:"
            echo "  sudo ./setup_usb_can_motors.sh"
            exit 1
        fi
        
        # Quick setup using slcan0 (USB-based)
        echo "   Using USB device: $SERIAL_DEVICE"
        pkill slcand 2>/dev/null || true
        sleep 1
        
        # Start slcand with 921600 baud serial rate
        if slcand -o -c -S 921600 -s6 "$SERIAL_DEVICE" $CAN_INTERFACE 2>/dev/null; then
            sleep 2
            ip link set $CAN_INTERFACE up 2>/dev/null || true
            echo "   ✅ USB-to-CAN interface ready ($CAN_INTERFACE)"
        else
            echo "   ❌ Failed to setup USB-to-CAN"
            echo ""
            echo "Please run full setup:"
            echo "  sudo ./setup_usb_can_motors.sh"
            exit 1
        fi
    fi
fi

# Check if test program exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "test_motors_12_14" ]; then
    echo "🔨 Compiling motor test..."
    
    if [ -f "test_motors_12_14.c" ]; then
        gcc -O2 -Wall -o test_motors_12_14 test_motors_12_14.c -lm
    elif [ -f "test_motor_12_14.c" ]; then
        gcc -O2 -Wall -o test_motors_12_14 test_motor_12_14.c -lm
    else
        echo "❌ Motor test source not found"
        exit 1
    fi
    
    chmod +x test_motors_12_14
    echo "   ✅ Compiled"
fi

# Verify USB-to-CAN is working
echo ""
echo "🔍 Verifying USB-to-CAN interface..."
if ip link show $CAN_INTERFACE | grep -q "UP"; then
    echo "   ✅ $CAN_INTERFACE is UP (USB-based CAN)"
    ip -details link show $CAN_INTERFACE | head -2 | sed 's/^/      /'
else
    echo "   ❌ $CAN_INTERFACE is not UP"
    exit 1
fi

echo ""
echo "🚀 Starting motor test..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the test
./test_motors_12_14

echo ""
echo "✅ Test complete!"
echo ""

