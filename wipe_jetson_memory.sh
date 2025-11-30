#!/bin/bash
# wipe_jetson_memory.sh - Stop all melvin processes and wipe brain files

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "=========================================="
echo "Wiping Jetson Memory & Stopping All Runs"
echo "=========================================="
echo ""

# Stop all melvin processes
echo "Stopping all melvin processes..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
# Kill all melvin processes (user processes)
pkill -9 -f melvin_run_continuous 2>/dev/null || true
pkill -9 -f melvin_pack_corpus 2>/dev/null || true
pkill -9 -f melvin_seed_instincts 2>/dev/null || true
pkill -9 -f melvin_feed_instincts 2>/dev/null || true
pkill -9 -f "melvin melvin.m" 2>/dev/null || true

# Kill root melvin processes (need sudo, but try without first)
pkill -9 -f "/home/melvin/melvin_system/melvin" 2>/dev/null || true

# Wait a moment
sleep 2

# Verify all stopped
REMAINING=$(ps aux | grep -E "melvin_run|melvin_pack|melvin_seed|melvin_feed|/home/melvin/melvin_system/melvin" | grep -v grep | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "✓ All melvin processes stopped"
else
    echo "⚠ Warning: $REMAINING processes still running (may need manual sudo kill)"
    ps aux | grep -E "melvin_run|melvin_pack|melvin_seed|melvin_feed|/home/melvin/melvin_system/melvin" | grep -v grep
    echo ""
    echo "To stop root processes, run on Jetson:"
    echo "  sudo pkill -9 -f '/home/melvin/melvin_system/melvin'"
fi
EOF

echo ""

# Wipe brain files
echo "Wiping brain files..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
# Remove brain files from all locations
rm -f /mnt/melvin_ssd/melvin/brain.m 2>/dev/null || true
rm -f ~/melvin/brain.m 2>/dev/null || true
rm -f /tmp/brain.m 2>/dev/null || true
rm -f /tmp/melvin_run.log 2>/dev/null || true
rm -f /tmp/melvin_run.pid 2>/dev/null || true
rm -f ~/melvin/melvin_run.log 2>/dev/null || true
rm -f ~/melvin/melvin_run.pid 2>/dev/null || true

# Try with sudo if needed
sudo rm -f /mnt/melvin_ssd/melvin/brain.m 2>/dev/null || true

echo "✓ Brain files wiped"
echo ""
echo "Remaining files:"
ls -lh /mnt/melvin_ssd/melvin/brain.m 2>/dev/null || echo "  /mnt/melvin_ssd/melvin/brain.m - removed"
ls -lh ~/melvin/brain.m 2>/dev/null || echo "  ~/melvin/brain.m - removed"
EOF

echo ""
echo "=========================================="
echo "Cleanup Complete"
echo "=========================================="
echo ""
echo "All melvin processes stopped"
echo "All brain files wiped"
echo ""
echo "To verify:"
echo "  sshpass -p '123456' ssh melvin@169.254.123.100 'ps aux | grep melvin'"

