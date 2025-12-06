#!/bin/bash
# start_display_persistent.sh - Start persistent X session

if [ "$EUID" -ne 0 ]; then 
    sudo "$0" "$@"
    exit $?
fi

echo "Starting persistent X session for jetson-io..."
echo ""

# Create xinitrc to keep X alive
cat > /tmp/xinitrc << 'XINITRC'
#!/bin/bash
# Keep X server alive and run jetson-io

# Set background
xsetroot -solid black 2>/dev/null

# Run jetson-io in xterm
xterm -geometry 120x40 -fa 'Monospace' -fs 12 -e bash -c '
echo "╔═══════════════════════════════════════════╗"
echo "║     Jetson CAN Configuration              ║"  
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "Running jetson-io..."
echo "Use arrow keys to navigate, SPACE to select, ENTER to confirm"
echo ""
sleep 2

/opt/nvidia/jetson-io/jetson-io.py

echo ""
echo "═══════════════════════════════════════════"
echo ""
echo "Did you enable CAN0 and save?"
echo "  - If YES: Jetson will reboot automatically"
echo "  - If NO: Press Ctrl+D to exit and try again"
echo ""
read -p "Press Enter to close (or Ctrl+C to keep X running)..."
' &

# Keep X alive - wait for xterm to finish
wait
XINITRC

chmod +x /tmp/xinitrc

echo "Starting X..."
startx /tmp/xinitrc -- :0 vt7 2>&1 &

echo ""
echo "═══════════════════════════════════════════"
echo "X server should be running on HDMI now!"
echo "═══════════════════════════════════════════"
echo ""
echo "Look at your monitor - you should see xterm"
echo ""
echo "To stop X: sudo killall X"
echo ""

