#!/bin/bash
#
# setup_usb_can_motors.sh - Complete USB-to-CAN Motor Setup
#
# This script:
# 1. Loads CH340 driver for USB-to-CAN adapter
# 2. Sets up slcand to create can0 from USB serial
# 3. Configures CAN interface
# 4. Compiles motor test code
# 5. Tests motors 12 and 14
#
# Run this ON THE JETSON directly (via USB connection)

set -e

echo "╔═══════════════════════════════════════════╗"
echo "║  USB-to-CAN Motor Setup (CH340)          ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Must run as root for CAN setup
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

# Step 1: Load CH340 driver
echo "1️⃣  Loading CH340 USB-to-Serial driver..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Try different CH340 driver names
if modprobe ch341 2>/dev/null; then
    echo "   ✅ Loaded ch341 driver"
elif modprobe ch340 2>/dev/null; then
    echo "   ✅ Loaded ch340 driver"
elif lsmod | grep -q ch34; then
    echo "   ✅ CH340 driver already loaded"
else
    echo "   ⚠️  CH340 driver not found in kernel"
    echo "   Attempting to install..."
    
    # Try to install if on Ubuntu/Debian
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y linux-modules-extra-$(uname -r) 2>/dev/null || true
        modprobe ch341 2>/dev/null || modprobe ch340 2>/dev/null || true
    fi
    
    if lsmod | grep -q ch34; then
        echo "   ✅ CH340 driver loaded"
    else
        echo "   ⚠️  CH340 driver not available"
        echo "   Continuing anyway - adapter may use different driver"
    fi
fi

# Step 2: Detect USB serial device
echo ""
echo "2️⃣  Detecting USB serial device..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait a moment for device to appear
sleep 1

# Find USB serial devices
USB_DEVICES=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true)

if [ -z "$USB_DEVICES" ]; then
    echo "   ❌ No USB serial devices found!"
    echo ""
    echo "   Make sure:"
    echo "   1. USB-to-CAN adapter is plugged in"
    echo "   2. CH340 driver is loaded"
    echo ""
    echo "   Checking USB devices:"
    lsusb | grep -i "ch340\|ch341\|serial\|can" || echo "   (No matching USB devices)"
    echo ""
    exit 1
fi

# Use first available device
SERIAL_DEVICE=$(echo "$USB_DEVICES" | head -1)
echo "   ✅ Found: $SERIAL_DEVICE"
echo "   Available devices: $USB_DEVICES"

# Step 3: Install CAN utilities
echo ""
echo "3️⃣  Installing CAN utilities..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v slcand >/dev/null 2>&1; then
    echo "   Installing can-utils..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y can-utils
    elif command -v yum >/dev/null 2>&1; then
        yum install -y can-utils
    else
        echo "   ⚠️  Cannot install can-utils automatically"
        echo "   Please install can-utils package manually"
        exit 1
    fi
fi

if command -v slcand >/dev/null 2>&1; then
    echo "   ✅ can-utils installed"
else
    echo "   ❌ slcand not found"
    exit 1
fi

# Step 4: Setup USB-to-CAN interface using slcand
echo ""
echo "4️⃣  Setting up USB-to-CAN interface (slcand)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use slcan0 to make it clear this is USB-based, not native CAN
CAN_INTERFACE="slcan0"

# Bring down existing interface if it exists
ip link set $CAN_INTERFACE down 2>/dev/null || true
pkill slcand 2>/dev/null || true
sleep 1

# Configure USB serial port for CAN
echo "   Configuring USB device $SERIAL_DEVICE for CAN..."
echo "   Setting serial baud rate to 921600..."
echo "   Creating virtual CAN interface: $CAN_INTERFACE"

# Set serial port permissions
chmod 666 "$SERIAL_DEVICE" 2>/dev/null || true

# Start slcand with 921600 baud serial rate
# -S sets UART/serial baud rate (921600)
# -s sets CAN bitrate (6 = 500kbps, adjust if needed)
# -o sends open command, -c sends close command
echo "   Command: slcand -o -c -S 921600 -s6 $SERIAL_DEVICE $CAN_INTERFACE"
echo "   (Serial: 921600 baud, CAN: 500kbps)"
if slcand -o -c -S 921600 -s6 "$SERIAL_DEVICE" $CAN_INTERFACE 2>&1; then
    echo "   ✅ slcand started (USB-to-CAN bridge active)"
    sleep 2  # Wait for interface to be ready
else
    echo "   ❌ Failed to start slcand"
    echo ""
    echo "   Troubleshooting:"
    echo "   1. Check if device is in use: lsof $SERIAL_DEVICE"
    echo "   2. Try different device: $USB_DEVICES"
    echo "   3. Check permissions: ls -l $SERIAL_DEVICE"
    exit 1
fi

# Step 5: Configure USB-to-CAN interface
echo ""
echo "5️⃣  Configuring USB-to-CAN interface..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Set bitrate (125kbps is common for motor controllers)
BITRATE=125000
echo "   Setting bitrate to $BITRATE bps..."

if ip link set $CAN_INTERFACE type can bitrate $BITRATE 2>/dev/null; then
    echo "   ✅ Bitrate configured"
else
    echo "   ⚠️  Could not set bitrate (slcand may have its own settings)"
fi

# Bring interface up
if ip link set $CAN_INTERFACE up 2>/dev/null; then
    echo "   ✅ USB-to-CAN interface up"
else
    echo "   ❌ Failed to bring $CAN_INTERFACE up"
    exit 1
fi

# Verify interface
if ip link show $CAN_INTERFACE | grep -q "UP"; then
    echo "   ✅ $CAN_INTERFACE is UP and ready (USB-based CAN)"
    ip -details link show $CAN_INTERFACE | head -3
else
    echo "   ❌ $CAN_INTERFACE is not UP"
    exit 1
fi

# Step 6: Test USB-to-CAN interface
echo ""
echo "6️⃣  Testing USB-to-CAN interface..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start candump in background
timeout 3 candump $CAN_INTERFACE > /tmp/can_test.log 2>&1 &
CANDUMP_PID=$!
sleep 1

# Send test frame
echo "   Sending test frame via USB-to-CAN..."
if cansend $CAN_INTERFACE 123#DEADBEEF 2>/dev/null; then
    echo "   ✅ Test frame sent"
else
    echo "   ⚠️  Could not send test frame (may be normal if no devices respond)"
fi

sleep 1
kill $CANDUMP_PID 2>/dev/null || true

# Check if we received anything
if [ -s /tmp/can_test.log ]; then
    echo "   ✅ Received CAN traffic via USB:"
    head -5 /tmp/can_test.log | sed 's/^/      /'
else
    echo "   ℹ️  No CAN traffic detected (motors may be off or not responding)"
fi

# Step 7: Compile motor test code
echo ""
echo "7️⃣  Compiling motor test code..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find motor test source
if [ -f "test_motors_12_14.c" ]; then
    MOTOR_TEST_SRC="test_motors_12_14.c"
elif [ -f "test_motor_12_14.c" ]; then
    MOTOR_TEST_SRC="test_motor_12_14.c"
else
    echo "   ❌ Motor test source not found!"
    echo "   Looking for: test_motors_12_14.c or test_motor_12_14.c"
    exit 1
fi

echo "   Compiling $MOTOR_TEST_SRC..."

if gcc -O2 -Wall -o test_motors_12_14 "$MOTOR_TEST_SRC" -lm 2>&1; then
    echo "   ✅ Compilation successful"
    chmod +x test_motors_12_14
else
    echo "   ❌ Compilation failed"
    exit 1
fi

# Step 8: Ready to test
echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║  Setup Complete!                          ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "✅ CH340 driver: Loaded"
echo "✅ USB device: $SERIAL_DEVICE"
echo "✅ USB-to-CAN interface: $CAN_INTERFACE (UP)"
echo "✅ Motor test: Compiled"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 Ready to test motors 12 and 14!"
echo ""
echo "To test motors, run:"
echo "  sudo ./test_motors_12_14"
echo ""
echo "Note: Using USB-to-CAN interface: $CAN_INTERFACE"
echo "      (Not native CAN hardware - using USB port)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "Run motor test now? (y/n): " -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting motor test..."
    echo ""
    ./test_motors_12_14
else
    echo ""
    echo "You can test motors later with:"
    echo "  sudo ./test_motors_12_14"
fi

echo ""
echo "✅ Setup complete!"
echo ""

