#!/bin/bash
# start_melvin_learning.sh - Start Melvin with proper brain and monitoring
# This starts melvin_run_continuous with the best available brain file

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "=========================================="
echo "Starting Melvin Learning Session"
echo "=========================================="
echo ""

# Check connection
if ! ping -c 1 -W 1 "$JETSON_IP" &> /dev/null; then
    echo "❌ ERROR: Cannot reach Jetson at $JETSON_IP"
    echo "   Check USB connection"
    exit 1
fi

echo "✅ Connected to Jetson"
echo ""

# Check what brain files exist and choose the best one
echo "Checking available brain files..."
BRAIN_CHOICE=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
cd ~/melvin
if [ -f brain_connected.m ]; then
    echo "brain_connected.m"
elif [ -f brain_preseeded.m ]; then
    echo "brain_preseeded.m"
elif [ -f brain_test.m ]; then
    echo "brain_test.m"
elif [ -f brain.m ]; then
    echo "brain.m"
else
    echo "none"
fi
EOF
)

if [ "$BRAIN_CHOICE" == "none" ]; then
    echo "❌ ERROR: No brain files found on Jetson"
    echo "   Create a brain first with create_brain.sh"
    exit 1
fi

echo "✅ Using brain: $BRAIN_CHOICE"
echo ""

# Check if already running
echo "Checking for existing processes..."
EXISTING=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "ps aux | grep melvin_run_continuous | grep -v grep | awk '{print \$2}'" 2>/dev/null)

if [ -n "$EXISTING" ]; then
    echo "⚠️  Melvin is already running (PID: $EXISTING)"
    read -p "Stop it and start fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
            "kill $EXISTING 2>/dev/null; sleep 2; echo '  ✓ Stopped'"
    else
        echo "Keeping existing process. Monitor with: ./jetson_live_monitor.sh"
        exit 0
    fi
fi

echo ""
echo "Starting Melvin continuous runner..."
echo ""

# Start melvin_run_continuous
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << EOF
cd ~/melvin

# Start in background, log to file
nohup ./melvin_run_continuous $BRAIN_CHOICE 1 > /tmp/melvin_run.log 2>&1 &
NEW_PID=\$!
echo \$NEW_PID > /tmp/melvin_run.pid

sleep 3

# Check if started
if ps -p \$NEW_PID > /dev/null 2>&1; then
    echo "✅ Melvin started successfully!"
    echo "   PID: \$NEW_PID"
    echo "   Brain: $BRAIN_CHOICE"
    echo "   Log: /tmp/melvin_run.log"
else
    echo "❌ Failed to start - check logs:"
    tail -20 /tmp/melvin_run.log
    exit 1
fi
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to start Melvin"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Melvin is now learning!"
echo "=========================================="
echo ""
echo "📊 Monitor progress:"
echo "   ./jetson_live_monitor.sh          # Live dashboard"
echo "   ./check_melvin_status.sh          # Quick status"
echo ""
echo "📝 View logs:"
echo "   sshpass -p '123456' ssh melvin@169.254.123.100 'tail -f /tmp/melvin_run.log'"
echo ""
echo "🛑 Stop Melvin:"
echo "   sshpass -p '123456' ssh melvin@169.254.123.100 'kill \$(cat /tmp/melvin_run.pid)'"
echo ""
echo "💡 What to watch for:"
echo "   - avg_activation increasing from 0.0"
echo "   - avg_chaos increasing from 0.0"
echo "   - Edge count growing"
echo "   - Output node activations (when learning progresses)"
echo ""

