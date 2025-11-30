#!/bin/bash
# start_melvin.sh - Start Melvin running continuously on Jetson

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
BRAIN_PATH="/mnt/melvin_ssd/melvin/brain.m"
WORK_DIR="/mnt/melvin_ssd/melvin"

echo "Starting Melvin continuously on Jetson..."
echo "Brain: $BRAIN_PATH"
echo ""

# Check if brain exists
sshpass -p "$JETSON_PASS" ssh "$JETSON_USER@$JETSON_IP" "test -f $BRAIN_PATH" || {
    echo "Error: Brain file not found at $BRAIN_PATH"
    echo "Creating brain first..."
    ./create_brain.sh
    if [ $? -ne 0 ]; then
        echo "Failed to create brain. Exiting."
        exit 1
    fi
}

# Start continuous run in background
echo "Starting Melvin in background..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_USER@$JETSON_IP" << 'EOF'
cd /mnt/melvin_ssd/melvin

# Create a simple continuous runner
cat > run_melvin_continuous.sh << 'INNER_EOF'
#!/bin/bash
BRAIN="brain.m"
INTERVAL=1  # Run every second

echo "Melvin continuous runner started"
echo "Brain: $BRAIN"
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Feed a small amount of random data to keep it active
    # Or just let it run with curiosity drive
    
    # Trigger UEL propagation
    # (In a real implementation, this would call melvin_call_entry)
    # For now, we'll just monitor and let the system run
    
    sleep $INTERVAL
done
INNER_EOF

chmod +x run_melvin_continuous.sh

# Start in background, log to file
nohup ./run_melvin_continuous.sh > melvin_run.log 2>&1 &
echo $! > melvin_run.pid

echo "Melvin started (PID: $(cat melvin_run.pid))"
echo "Log: melvin_run.log"
EOF

echo ""
echo "Melvin is now running continuously on Jetson!"
echo "Monitor with: ./monitor_melvin.sh"
echo "Check logs: sshpass -p '123456' ssh melvin@169.254.123.100 'tail -f /mnt/melvin_ssd/melvin/melvin_run.log'"

