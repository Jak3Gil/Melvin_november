#!/bin/bash
# start_jetson_display.sh - Start minimal X session on Jetson
# Run this ON the Jetson via HDMI monitor

echo "╔═══════════════════════════════════════════╗"
echo "║  Starting Minimal Display Session         ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

echo "1️⃣  Stopping any existing display managers..."
systemctl stop gdm3 2>/dev/null
systemctl stop lightdm 2>/dev/null

echo ""
echo "2️⃣  Starting X server on display :0..."

# Start X server in background
startx &
XPID=$!

sleep 3

echo ""
echo "3️⃣  Display should now be active on HDMI"
echo ""
echo "Opening terminal and jetson-io..."
echo ""

# Set display
export DISPLAY=:0

# Start xterm with jetson-io
DISPLAY=:0 xterm -e "
echo '═══════════════════════════════════════════'
echo 'Jetson CAN Configuration'
echo '═══════════════════════════════════════════'
echo ''
echo 'Press Enter to run jetson-io...'
read
sudo /opt/nvidia/jetson-io/jetson-io.py
echo ''
echo 'After enabling CAN0:'
echo '  1. Save configuration'
echo '  2. Exit this terminal'
echo '  3. Reboot Jetson'
echo ''
echo 'Press Enter to close...'
read
" &

echo ""
echo "═══════════════════════════════════════════"
echo "Instructions:"
echo "═══════════════════════════════════════════"
echo ""
echo "1. Look at your HDMI monitor"
echo "2. You should see xterm with jetson-io"
echo "3. Use arrow keys to navigate"
echo "4. Enable 'CAN0' or 'mttcan'"
echo "5. Save and reboot"
echo ""
echo "Press Ctrl+C here when done"
echo ""

wait

