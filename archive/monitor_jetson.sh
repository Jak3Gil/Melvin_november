#!/bin/bash
# Monitor Melvin running on Jetson remotely

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${1:-jetson.local}"
JETSON_DIR="~/melvin"
BRAIN_FILE="${JETSON_DIR}/melvin.m"
REFRESH_RATE="${2:-0.5}"

echo "=========================================="
echo "Monitoring Melvin on Jetson"
echo "=========================================="
echo "Jetson: ${JETSON_USER}@${JETSON_HOST}"
echo "Brain: ${BRAIN_FILE}"
echo "Refresh: ${REFRESH_RATE}s"
echo "Press Ctrl+C to stop"
echo ""

# Check if monitor exists on Jetson, if not, run it remotely
if sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" \
    "test -f ${JETSON_DIR}/monitor_melvin" 2>/dev/null; then
    # Run monitor on Jetson and display output
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -t "${JETSON_USER}@${JETSON_HOST}" \
        "cd ${JETSON_DIR} && ./monitor_melvin.sh ${BRAIN_FILE} ${REFRESH_RATE}"
else
    # Fallback: use local monitor if we can copy brain file
    echo "Monitor not found on Jetson, using local monitor..."
    echo "Copying brain file temporarily..."
    
    TEMP_BRAIN="/tmp/melvin_remote.m"
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        "${JETSON_USER}@${JETSON_HOST}:${BRAIN_FILE}" "$TEMP_BRAIN" 2>/dev/null
    
    if [ -f "$TEMP_BRAIN" ]; then
        ./monitor_melvin.sh "$TEMP_BRAIN" "$REFRESH_RATE" &
        MONITOR_PID=$!
        
        # Keep copying brain file periodically
        while kill -0 $MONITOR_PID 2>/dev/null; do
            sleep "$REFRESH_RATE"
            sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
                "${JETSON_USER}@${JETSON_HOST}:${BRAIN_FILE}" "$TEMP_BRAIN" 2>/dev/null
        done
        
        rm -f "$TEMP_BRAIN"
    else
        echo "ERROR: Cannot access brain file on Jetson"
        exit 1
    fi
fi

