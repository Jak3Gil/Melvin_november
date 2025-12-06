#!/bin/bash
# enable_can_with_display.sh - Start X and run jetson-io
# Run this ON the Jetson (will show on HDMI monitor)

if [ "$EUID" -ne 0 ]; then 
    sudo "$0" "$@"
    exit $?
fi

echo "╔═══════════════════════════════════════════╗"
echo "║  Starting Display for CAN Configuration   ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

echo "Connect HDMI monitor now if not already connected"
echo "Press Enter when ready..."
read

echo ""
echo "Starting X server..."

# Kill any existing X
killall X Xorg 2>/dev/null

# Start X on display :0
X :0 &
XPID=$!

sleep 3

# Set display environment
export DISPLAY=:0

echo ""
echo "✅ X server started (PID: $XPID)"
echo "   Check your HDMI monitor - should see blank screen"
echo ""

sleep 2

echo "Starting xterm with jetson-io..."
echo ""

# Run xterm with jetson-io in it
DISPLAY=:0 xterm -geometry 100x40 -fa 'Monospace' -fs 12 -e bash -c '
clear
echo "╔═══════════════════════════════════════════╗"
echo "║        Jetson CAN Configuration           ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "You are now in xterm on the HDMI display."
echo ""
echo "Steps:"
echo "  1. Press Enter to start jetson-io"
echo "  2. Use arrow keys to navigate"
echo "  3. Select \"Configure Jetson 40pin Header\""
echo "  4. Find and enable \"CAN0\" or \"mttcan\""
echo "  5. Press SPACE to check the box"
echo "  6. Select \"Save and reboot\""
echo "  7. Wait for reboot"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "Starting jetson-io (use arrow keys + space + enter)..."
echo ""
sleep 2

sudo /opt/nvidia/jetson-io/jetson-io.py

echo ""
echo "═══════════════════════════════════════════"
echo ""
echo "If you enabled CAN and selected \"Save and reboot\","
echo "the Jetson will reboot automatically."
echo ""
echo "After reboot, the CAN pins will be enabled!"
echo ""
echo "Press Enter to close this terminal..."
read
' &

echo "═══════════════════════════════════════════"
echo "Look at your HDMI monitor now!"
echo "═══════════════════════════════════════════"
echo ""
echo "You should see a terminal window with instructions."
echo ""
echo "Press Ctrl+C here when you're done with jetson-io"
echo ""

# Wait for X session
wait $XPID

echo ""
echo "Display session ended."
echo "If you rebooted, CAN should now be enabled!"

