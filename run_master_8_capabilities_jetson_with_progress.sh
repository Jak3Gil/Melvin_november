#!/bin/bash

# Test Runner for test_master_8_capabilities on Jetson with Progress Bar
# Shows real-time progress so user knows test is running

set -e

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="169.254.123.100"
JETSON_PATH="/home/melvin/melvin_tests"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "MASTER TEST SUITE — 8 Core Capabilities"
echo "JETSON RUNNER (with Progress Monitoring)"
echo "=========================================="
echo ""

# Check connection
echo -n "Checking connection... "
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo -e "${RED}FAILED${NC}"
    echo "If using USB/COM8 serial, ensure SSH tunnel is set up"
    exit 1
fi
echo -e "${GREEN}OK${NC}"
echo ""

# Copy files
echo -n "[1/4] Copying files to Jetson... "
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    test_master_8_capabilities.c \
    melvin.c \
    instincts.c \
    test_helpers.h \
    melvin_instincts.h \
    melvin_exec_helpers.c \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/ > /dev/null 2>&1
echo -e "${GREEN}✓${NC}"
echo ""

# Compile
echo -n "[2/4] Compiling on Jetson... "
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && gcc -o test_master_8_capabilities test_master_8_capabilities.c -lm -std=c11 -Wall -O0 2>&1 | tail -3" > /tmp/jetson_compile.log 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}FAILED${NC}"
    cat /tmp/jetson_compile.log
    exit 1
fi
echo -e "${GREEN}✓${NC}"
echo ""

# Run test with progress monitoring
echo "[3/4] Running test suite (this may take several minutes)..."
echo ""
echo "Progress will be shown below. Test is running if you see updates."
echo ""

# Create a progress monitoring function
monitor_progress() {
    local test_pid=$1
    local last_size=0
    local no_change_count=0
    local dots=0
    
    while kill -0 $test_pid 2>/dev/null; do
        # Check log file size
        local current_size=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
            "stat -f%z $JETSON_PATH/results/test_master_8_capabilities.log 2>/dev/null || echo 0")
        
        if [ "$current_size" != "$last_size" ]; then
            # Log is growing - test is active
            echo -ne "\r${BLUE}[RUNNING]${NC} Test is active... (log size: ${current_size} bytes)   "
            last_size=$current_size
            no_change_count=0
            dots=0
        else
            # Log not growing - show heartbeat
            no_change_count=$((no_change_count + 1))
            dots=$((dots + 1))
            if [ $dots -gt 3 ]; then
                dots=0
            fi
            local dot_str=$(printf '.%.0s' $(seq 1 $dots))
            echo -ne "\r${YELLOW}[WAITING]${NC} Processing${dot_str}   "
            
            # If no change for 30 seconds, show warning
            if [ $no_change_count -gt 30 ]; then
                echo -ne "\r${YELLOW}[SLOW]${NC} Test may be in long-running phase (capability 8: 1000 ticks)...   "
            fi
        fi
        
        sleep 1
    done
    echo -e "\r${GREEN}[COMPLETE]${NC} Test finished!                                    "
}

# Run test in background and monitor
echo "Starting test execution..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && ./test_master_8_capabilities > results/test_master_8_capabilities.log 2>&1" &
TEST_PID=$!

# Monitor progress
monitor_progress $TEST_PID

# Wait for test to complete
wait $TEST_PID
TEST_RESULT=$?

echo ""
echo "[4/4] Downloading results... "

# Download results
mkdir -p jetson_test_results
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_master_8_capabilities.log \
    jetson_test_results/ 2>/dev/null || true

# Show summary
echo ""
echo "=========================================="
echo "TEST RESULTS SUMMARY"
echo "=========================================="
echo ""

if [ -f jetson_test_results/test_master_8_capabilities.log ]; then
    # Show last 100 lines (summary section)
    tail -100 jetson_test_results/test_master_8_capabilities.log | grep -A 50 "TEST SUMMARY\|TEST RESULTS\|Capability\|PASS\|FAIL" | head -80
    
    # Show EXEC stats if available
    echo ""
    echo "--- EXEC Statistics ---"
    grep -E "EXEC STATS|exec_attempts|exec_executed|exec_skipped" jetson_test_results/test_master_8_capabilities.log | tail -10 || echo "No EXEC stats found"
else
    echo "Could not download test log"
fi

echo ""
echo "=========================================="
echo "FULL LOG: jetson_test_results/test_master_8_capabilities.log"
echo "=========================================="
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST SUITE COMPLETED SUCCESSFULLY${NC}"
    exit 0
else
    echo -e "${RED}✗ TEST SUITE FAILED OR HUNG${NC}"
    exit 1
fi

