#!/bin/bash
# jetson_live_monitor.sh - Continuous monitoring of Jetson status
# Auto-refreshes every few seconds to show what's happening

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
REFRESH_INTERVAL=3

echo "=========================================="
echo "Live Monitoring: Jetson via USB"
echo "=========================================="
echo "Refresh interval: ${REFRESH_INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "JETSON LIVE STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if Jetson is reachable
    if ! ping -c 1 -W 1 "$JETSON_IP" &> /dev/null; then
        echo "❌ ERROR: Cannot reach Jetson at $JETSON_IP"
        echo "   Check USB connection"
        sleep $REFRESH_INTERVAL
        continue
    fi
    
    echo "✅ Connection: Active"
    echo ""
    
    # Running processes
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "RUNNING PROCESSES:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    PROCESSES=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "ps aux | grep -E '(melvin|start_continuous)' | grep -v grep" 2>/dev/null)
    
    if [ -z "$PROCESSES" ]; then
        echo "  No Melvin processes running"
    else
        echo "$PROCESSES" | awk '{printf "  PID %-6s  CPU: %-5s  MEM: %-5s  CMD: %s\n", $2, $3"%", $4"%", $11}'
    fi
    echo ""
    
    # Latest logs
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "LATEST ACTIVITY (last 5 lines):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    LOGS=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "tail -5 /tmp/melvin_run.log 2>/dev/null")
    
    if [ -z "$LOGS" ]; then
        echo "  No activity logged"
    else
        echo "$LOGS" | sed 's/^/  /'
    fi
    echo ""
    
    # Brain file status
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "BRAIN FILE:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    BRAIN_INFO=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "ls -lh ~/melvin/brain.m 2>/dev/null | awk '{print \$5, \$6, \$7, \$8}'")
    
    if [ -z "$BRAIN_INFO" ]; then
        echo "  Brain file not found"
    else
        echo "  Size & Modified: $BRAIN_INFO"
    fi
    echo ""
    
    # System resources
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "SYSTEM RESOURCES:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    RESOURCES=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "free -h | grep Mem; uptime" 2>/dev/null)
    
    if [ -n "$RESOURCES" ]; then
        MEM_LINE=$(echo "$RESOURCES" | head -1 | awk '{printf "  Memory: %s used / %s total", $3, $2}')
        UPTIME_LINE=$(echo "$RESOURCES" | tail -1 | awk -F'up ' '{print $2}' | awk -F', ' '{print "  Uptime: "$1}')
        echo "$MEM_LINE"
        echo "$UPTIME_LINE"
    fi
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Next refresh in ${REFRESH_INTERVAL}s..."
    
    sleep $REFRESH_INTERVAL
done

