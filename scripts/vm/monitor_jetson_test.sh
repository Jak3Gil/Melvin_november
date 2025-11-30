#!/bin/bash

# Real-time test progress monitor for Jetson
# Shows which capability is currently running

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="169.254.123.100"
JETSON_PATH="/home/melvin/melvin_tests"

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "Monitoring test progress on Jetson..."
echo "Press Ctrl+C to stop monitoring (test will continue running)"
echo ""

last_line=""
last_capability=""
line_count=0

while true; do
    # Get last few lines of log
    current_lines=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
        "tail -20 $JETSON_PATH/results/test_master_8_capabilities.log 2>/dev/null || echo ''")
    
    current_line_count=$(echo "$current_lines" | wc -l)
    current_last_line=$(echo "$current_lines" | tail -1)
    
    # Check if log is growing
    if [ "$current_line_count" != "$line_count" ] || [ "$current_last_line" != "$last_line" ]; then
        # Extract current capability
        capability=$(echo "$current_lines" | grep -E "CAPABILITY [0-9]|Capability [0-9]" | tail -1 | sed 's/.*CAPABILITY \([0-9]\):.*/\1/' | sed 's/.*Capability \([0-9]\):.*/\1/')
        
        if [ -n "$capability" ] && [ "$capability" != "$last_capability" ]; then
            echo -e "${GREEN}[PROGRESS]${NC} Now running Capability $capability"
            last_capability=$capability
        fi
        
        # Show test case results if available
        case_result=$(echo "$current_lines" | grep -E "Case [0-9]+:|✓|✗" | tail -1)
        if [ -n "$case_result" ]; then
            echo "  $case_result"
        fi
        
        # Show EXEC stats if available
        exec_stats=$(echo "$current_lines" | grep -E "EXEC STATS|exec_attempts|exec_executed" | tail -1)
        if [ -n "$exec_stats" ]; then
            echo -e "  ${BLUE}$exec_stats${NC}"
        fi
        
        line_count=$current_line_count
        last_line=$current_last_line
    else
        # Show heartbeat
        echo -ne "\r${YELLOW}[WAITING]${NC} Test running... (no new output)    "
    fi
    
    # Check if test process is still running
    if ! sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
        "pgrep -f test_master_8_capabilities > /dev/null" 2>/dev/null; then
        echo ""
        echo -e "${GREEN}[COMPLETE]${NC} Test process finished!"
        break
    fi
    
    sleep 2
done

echo ""
echo "Downloading final results..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_master_8_capabilities.log \
    jetson_test_results/ 2>/dev/null

echo ""
echo "Final log saved to: jetson_test_results/test_master_8_capabilities.log"

