#!/bin/bash
# start_melvin_continuous.sh - Start Melvin running continuously

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "Starting Melvin continuously on Jetson..."
echo ""

sshpass -p "$JETSON_PASS" ssh "$JETSON_USER@$JETSON_IP" << 'EOF'
cd /tmp

# Copy brain if not exists
if [ ! -f brain.m ]; then
    echo "Copying brain from SSD..."
    cp /mnt/melvin_ssd/melvin/brain.m . || {
        echo "Error: Brain file not found. Create it first with create_brain.sh"
        exit 1
    }
fi

# Check if already running
if [ -f melvin_run.pid ]; then
    PID=$(cat melvin_run.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Melvin is already running (PID: $PID)"
        echo "Stop it first with: kill $PID"
        exit 1
    fi
fi

# Start Melvin
echo "Starting Melvin continuous runner..."
nohup /mnt/melvin_ssd/melvin/melvin_run_continuous brain.m 1 > melvin_run.log 2>&1 &
echo $! > melvin_run.pid

sleep 2

# Check if started
if ps -p $(cat melvin_run.pid) > /dev/null 2>&1; then
    echo "Melvin started successfully!"
    echo "PID: $(cat melvin_run.pid)"
    echo "Log: /tmp/melvin_run.log"
    echo ""
    echo "Monitor with: ./monitor_melvin.sh"
    echo "View logs: sshpass -p '123456' ssh melvin@169.254.123.100 'tail -f /tmp/melvin_run.log'"
else
    echo "Error: Failed to start Melvin"
    echo "Check log: /tmp/melvin_run.log"
    exit 1
fi
EOF

echo ""
echo "Melvin is now running continuously!"
echo "Use ./monitor_melvin.sh to watch it live"

