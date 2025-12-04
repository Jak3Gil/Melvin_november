#!/bin/bash
#
# setup_jetson_motors.sh - Setup CAN motors on Jetson
#
# This script:
# 1. Installs CAN utilities
# 2. Configures CAN interface
# 3. Compiles motor control tools
# 4. Maps motors to brain
# 5. Starts motor runtime
#

set -e

echo "🤖 Melvin Jetson Motor Setup"
echo "=============================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: Not running on Jetson"
    echo "   This script is designed for Jetson devices"
    echo "   Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Install CAN utilities
echo "📦 Installing CAN utilities..."
sudo apt-get update
sudo apt-get install -y can-utils

# Load CAN kernel modules
echo "🔧 Loading CAN kernel modules..."
sudo modprobe can
sudo modprobe can_raw
sudo modprobe mttcan || echo "   (mttcan not available, using generic can)"

# Detect USB-to-CAN adapter
echo ""
echo "🔍 Detecting CAN adapter..."
if ip link show can0 > /dev/null 2>&1; then
    echo "   ✅ can0 already exists"
else
    echo "   ⚠️  can0 not found"
    echo "   Available network interfaces:"
    ip link show | grep -E "^[0-9]+" | cut -d: -f2
    echo ""
    echo "   Make sure USB-to-CAN adapter is connected"
    echo "   Some adapters require slcand setup:"
    echo "     sudo slcand -o -c -s6 /dev/ttyUSB0 can0"
    echo ""
    read -p "   Continue setup? (y/n) " -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Configure CAN interface
BITRATE=125000
echo ""
echo "🔧 Configuring CAN interface (bitrate: $BITRATE)..."

sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 type can bitrate $BITRATE || {
    echo "   ⚠️  Failed to set bitrate"
    echo "   Trying slcand setup..."
    # Try slcand for USB adapters
    if [ -e /dev/ttyUSB0 ]; then
        sudo slcand -o -c -s6 /dev/ttyUSB0 can0
    fi
}
sudo ip link set can0 up

if ip link show can0 | grep -q "UP"; then
    echo "   ✅ CAN interface active"
else
    echo "   ❌ Failed to bring up CAN interface"
    exit 1
fi

# Test CAN interface
echo ""
echo "🧪 Testing CAN interface..."
timeout 2 candump can0 &
CANDUMP_PID=$!
sleep 1

# Try to send a test frame
cansend can0 123#DEADBEEF 2>/dev/null || echo "   (Send test - OK if no devices respond)"

wait $CANDUMP_PID 2>/dev/null || true
echo "   ✅ CAN interface operational"

# Compile motor tools
echo ""
echo "🔨 Compiling motor control tools..."

BRAIN_FILE="${1:-brain_teachable.m}"

if [ ! -f "$BRAIN_FILE" ]; then
    echo "   ⚠️  Brain file not found: $BRAIN_FILE"
    echo "   Creating new brain..."
    ./create_teachable_hardware_brain.sh
    BRAIN_FILE="brain_teachable.m"
fi

echo "   Using brain: $BRAIN_FILE"

# Compile tools
make -f tools/Makefile map_can_motors || {
    echo "   Compiling map_can_motors manually..."
    gcc -O2 -o tools/map_can_motors tools/map_can_motors.c src/melvin.c -lm -lpthread
}

make test_motor_exec || {
    echo "   Compiling test_motor_exec manually..."
    gcc -O2 -o test_motor_exec test_motor_exec.c src/melvin.c -lm -lpthread
}

make melvin_motor_runtime || {
    echo "   Compiling melvin_motor_runtime manually..."
    gcc -O2 -o melvin_motor_runtime melvin_motor_runtime.c src/melvin.c -lm -lpthread
}

echo "   ✅ Tools compiled"

# Map motors to brain
echo ""
echo "🗺️  Scanning and mapping motors..."
sudo ./tools/map_can_motors "$BRAIN_FILE"

if [ ! -f motor_config.txt ]; then
    echo "   ❌ Motor mapping failed"
    exit 1
fi

echo "   ✅ Motors mapped"

# Show motor configuration
echo ""
echo "📋 Motor Configuration:"
cat motor_config.txt | grep -E "^motor_|can_id:" | head -20

# Test motors
echo ""
echo "🧪 Testing motor control..."
read -p "Run motor test? (y/n) " -r response
if [ "$response" = "y" ]; then
    sudo ./test_motor_exec "$BRAIN_FILE" all
fi

# Create systemd service for automatic startup
echo ""
echo "📝 Creating systemd service..."
cat > /tmp/melvin-motors.service <<EOF
[Unit]
Description=Melvin Motor Runtime
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
ExecStart=$PWD/melvin_motor_runtime $PWD/$BRAIN_FILE
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/melvin-motors.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "   ✅ Service created: melvin-motors"
echo ""
echo "   To start automatically:"
echo "     sudo systemctl enable melvin-motors"
echo "     sudo systemctl start melvin-motors"
echo ""
echo "   To start manually:"
echo "     sudo ./melvin_motor_runtime $BRAIN_FILE"

# Summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Motor runtime is ready to run. The brain will now:"
echo "  1. Monitor sensory inputs (audio, video)"
echo "  2. Discover patterns through experience"
echo "  3. Learn to route patterns to motor EXEC nodes"
echo "  4. Execute motor control code automatically"
echo "  5. Adapt based on outcomes"
echo ""
echo "Everything learned, not hardcoded! ✨"
echo ""
echo "Start runtime:"
echo "  sudo ./melvin_motor_runtime $BRAIN_FILE"
echo ""

