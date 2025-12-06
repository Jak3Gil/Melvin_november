#!/bin/bash
# run_tests_via_usb.sh - Run complex tests on Jetson via USB connection
# 
# Uses USB networking (RNDIS) to connect to Jetson
# Falls back to serial if network unavailable

set -e

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
WORK_DIR="/home/melvin/melvin"
RESULTS_DIR="./jetson_test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "COMPLEX JETSON TEST SUITE (USB)"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Target: $JETSON_USER@$JETSON_IP (USB)"
echo ""

# Check USB device
echo "Checking USB connection..."
if [ -e /dev/tty.usbmodem* ] || [ -e /dev/ttyUSB* ]; then
    echo -e "${GREEN}✓ USB device detected${NC}"
    ls -la /dev/tty.usbmodem* /dev/ttyUSB* 2>/dev/null | head -3
else
    echo -e "${YELLOW}⚠ No USB serial device found${NC}"
fi
echo ""

# Check network connection
echo "Checking network connection..."
if ping -c 1 -W 2 $JETSON_IP > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Network connection OK${NC}"
    CONNECTION_METHOD="ssh"
else
    echo -e "${RED}✗ Network connection failed${NC}"
    echo -e "${YELLOW}Trying serial connection...${NC}"
    CONNECTION_METHOD="serial"
fi
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run command via SSH
run_ssh() {
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "$JETSON_USER@$JETSON_IP" "$@"
}

# Function to transfer file via SCP
transfer_file() {
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "$1" "$JETSON_USER@$JETSON_IP:$2"
}

# Transfer source files
echo "Transferring source files to Jetson..."
transfer_file "src/melvin.c" "$WORK_DIR/src/melvin.c" || {
    echo -e "${RED}ERROR: Failed to transfer melvin.c${NC}"
    exit 1
}
transfer_file "src/melvin.h" "$WORK_DIR/src/melvin.h" || {
    echo -e "${RED}ERROR: Failed to transfer melvin.h${NC}"
    exit 1
}
# Also copy to root
run_ssh "cd $WORK_DIR && cp src/melvin.c melvin.c && cp src/melvin.h melvin.h" 2>/dev/null || true
echo -e "${GREEN}✓ Source files transferred${NC}"
echo ""

# Transfer test files
echo "Transferring test files..."
TEST_FILES=(
    "evaluate_melvin_metrics.c"
    "test_exec_tracking.c"
    "test_prediction_signals.c"
    "test_unified_architecture.c"
    "real_exec_bridge.c"
)

for test_file in "${TEST_FILES[@]}"; do
    if [ -f "$test_file" ]; then
        transfer_file "$test_file" "$WORK_DIR/" && \
            echo -e "${GREEN}  ✓ $test_file${NC}" || \
            echo -e "${YELLOW}  ⚠ $test_file (failed)${NC}"
    fi
done
echo ""

# Run tests
echo "=========================================="
echo "RUNNING TESTS ON JETSON"
echo "=========================================="
echo ""

# Test 1: Evaluation Metrics
echo -e "${BLUE}TEST 1: Evaluation Metrics (5-domain)${NC}"
run_ssh << 'ENDSSH'
cd /home/melvin/melvin
mkdir -p evaluation_results
echo "Compiling evaluate_melvin_metrics..."
if [ -f melvin.c ]; then
    gcc -o evaluate_melvin_metrics evaluate_melvin_metrics.c real_exec_bridge.c melvin.c -I. -std=c11 -lm -O2 2>&1 | head -30
else
    gcc -o evaluate_melvin_metrics evaluate_melvin_metrics.c real_exec_bridge.c src/melvin.c -Isrc -I. -std=c11 -lm -O2 2>&1 | head -30
fi
if [ $? -eq 0 ]; then
    echo "Creating test brain file..."
    # Create a simple brain file using melvin_open (via a small C program)
    cat > create_test_brain.c << 'EOFCREATE'
#include <stdio.h>
#include "melvin.h"
int main() {
    Graph *g = melvin_open("test_brain.m", 10000, 50000, 1024*1024);
    if (g) {
        melvin_sync(g);
        melvin_close(g);
        printf("Brain created\n");
        return 0;
    }
    return 1;
}
EOFCREATE
    if [ -f melvin.c ]; then
        gcc -o create_test_brain create_test_brain.c melvin.c -I. -std=c11 -lm -O2 2>&1 | head -10
    else
        gcc -o create_test_brain create_test_brain.c src/melvin.c -Isrc -I. -std=c11 -lm -O2 2>&1 | head -10
    fi
    if [ -f create_test_brain ]; then
        ./create_test_brain
        BRAIN_FILE="test_brain.m"
    else
        BRAIN_FILE="brain.m"  # Use existing if available
    fi
    
    echo "Running evaluation metrics test with brain: $BRAIN_FILE..."
    if [ -f "$BRAIN_FILE" ]; then
        ./evaluate_melvin_metrics "$BRAIN_FILE"
        EXIT_CODE=$?
    else
        echo "ERROR: No brain file found"
        EXIT_CODE=1
    fi
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST1_RESULT=$?
if [ $TEST1_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 1 PASSED${NC}"
else
    echo -e "${RED}✗ TEST 1 FAILED${NC}"
fi
echo ""

# Test 2: EXEC Tracking
echo -e "${BLUE}TEST 2: EXEC Tracking${NC}"
run_ssh << 'ENDSSH'
cd /home/melvin/melvin
mkdir -p evaluation_results
echo "Compiling test_exec_tracking..."
if [ -f melvin.c ]; then
    gcc -o test_exec_tracking test_exec_tracking.c real_exec_bridge.c melvin.c -I. -std=c11 -lm -O2 2>&1 | head -30
else
    gcc -o test_exec_tracking test_exec_tracking.c real_exec_bridge.c src/melvin.c -Isrc -I. -std=c11 -lm -O2 2>&1 | head -30
fi
if [ $? -eq 0 ]; then
    echo "Running EXEC tracking test..."
    ./test_exec_tracking
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    if [ -f evaluation_results/exec_creation.log ]; then
        echo "✓ exec_creation.log created"
        cat evaluation_results/exec_creation.log
    fi
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST2_RESULT=$?
if [ $TEST2_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 2 PASSED${NC}"
else
    echo -e "${RED}✗ TEST 2 FAILED${NC}"
fi
echo ""

# Test 3: Prediction Signals
echo -e "${BLUE}TEST 3: Prediction Signals Sanity Check${NC}"
run_ssh << 'ENDSSH'
cd /home/melvin/melvin
mkdir -p evaluation_results
echo "Compiling test_prediction_signals..."
if [ -f melvin.c ]; then
    gcc -o test_prediction_signals test_prediction_signals.c real_exec_bridge.c melvin.c -I. -std=c11 -lm -O2 2>&1 | head -30
else
    gcc -o test_prediction_signals test_prediction_signals.c real_exec_bridge.c src/melvin.c -Isrc -I. -std=c11 -lm -O2 2>&1 | head -30
fi
if [ $? -eq 0 ]; then
    echo "Running prediction signals test..."
    ./test_prediction_signals test_brain.m 2>&1 | tee prediction_signals_output.txt
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST3_RESULT=$?
if [ $TEST3_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 3 PASSED${NC}"
else
    echo -e "${RED}✗ TEST 3 FAILED${NC}"
fi
echo ""

# Test 4: Unified Architecture Soak Test (REAL DATA)
echo -e "${BLUE}TEST 4: Unified Architecture Soak Test (Real Camera Data)${NC}"
run_ssh << 'ENDSSH'
cd /home/melvin/melvin
mkdir -p evaluation_results
echo "Compiling test_unified_architecture..."
if [ -f melvin.c ]; then
    gcc -o test_unified_architecture test_unified_architecture.c real_exec_bridge.c melvin.c -I. -std=c11 -lm -O2 2>&1 | head -30
else
    gcc -o test_unified_architecture test_unified_architecture.c real_exec_bridge.c src/melvin.c -Isrc -I. -std=c11 -lm -O2 2>&1 | head -30
fi
if [ $? -eq 0 ]; then
    echo "Running unified architecture test with REAL CAMERA DATA..."
    echo "Duration: 60 seconds (or 2000 steps if camera unavailable)"
    echo ""
    
    # Try camera first, fallback to synthetic if unavailable
    if [ -e /dev/video0 ] || [ -e /dev/video1 ]; then
        CAMERA_DEV=$(ls /dev/video* 2>/dev/null | head -1)
        echo "Using camera: $CAMERA_DEV"
        ./test_unified_architecture --seconds 60 --camera --device "$CAMERA_DEV" --brain unified_test_brain.m --verbose 2>&1 | tee unified_test_output.txt
    else
        echo "No camera found, using synthetic input for 2000 steps"
        ./test_unified_architecture --steps 2000 --no-camera --brain unified_test_brain.m --verbose 2>&1 | tee unified_test_output.txt
    fi
    
    EXIT_CODE=$?
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "=== Generated Datasets ==="
    if [ -f unified_metrics.csv ]; then
        LINES=$(wc -l < unified_metrics.csv)
        echo "✓ unified_metrics.csv: $LINES lines"
        echo "  First few lines:"
        head -5 unified_metrics.csv
    fi
    if [ -f unified_node_samples.csv ]; then
        LINES=$(wc -l < unified_node_samples.csv)
        echo "✓ unified_node_samples.csv: $LINES lines"
        echo "  First few lines:"
        head -5 unified_node_samples.csv
    fi
    echo ""
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST4_RESULT=$?
if [ $TEST4_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 4 PASSED${NC}"
else
    echo -e "${RED}✗ TEST 4 FAILED${NC}"
fi
echo ""

# Download results
echo "Downloading results..."
run_ssh "cd $WORK_DIR && ls -lh *.log *.csv evaluation_results/*.log evaluation_results/*.csv 2>/dev/null | head -30" || true

transfer_file "$JETSON_USER@$JETSON_IP:$WORK_DIR/*.log" "$RESULTS_DIR/" 2>/dev/null || true
transfer_file "$JETSON_USER@$JETSON_IP:$WORK_DIR/evaluation_results/*" "$RESULTS_DIR/" 2>/dev/null || true
transfer_file "$JETSON_USER@$JETSON_IP:$WORK_DIR/prediction_signals_output.txt" "$RESULTS_DIR/" 2>/dev/null || true

# Download unified test datasets
echo "Downloading unified test datasets..."
run_ssh "cd $WORK_DIR && ls -lh unified_*.csv unified_*.txt 2>/dev/null" || true
transfer_file "$WORK_DIR/unified_metrics.csv" "$RESULTS_DIR/unified_metrics.csv" 2>/dev/null && echo "✓ Downloaded unified_metrics.csv" || echo "⚠ unified_metrics.csv not found"
transfer_file "$WORK_DIR/unified_node_samples.csv" "$RESULTS_DIR/unified_node_samples.csv" 2>/dev/null && echo "✓ Downloaded unified_node_samples.csv" || echo "⚠ unified_node_samples.csv not found"
transfer_file "$WORK_DIR/unified_test_output.txt" "$RESULTS_DIR/unified_test_output.txt" 2>/dev/null && echo "✓ Downloaded unified_test_output.txt" || echo "⚠ unified_test_output.txt not found"

# Copy to current directory for processing
if [ -f "$RESULTS_DIR/unified_metrics.csv" ]; then
    cp "$RESULTS_DIR/unified_metrics.csv" . 2>/dev/null && echo "✓ Copied unified_metrics.csv to current directory"
fi
if [ -f "$RESULTS_DIR/unified_node_samples.csv" ]; then
    cp "$RESULTS_DIR/unified_node_samples.csv" . 2>/dev/null && echo "✓ Copied unified_node_samples.csv to current directory"
fi

echo -e "${GREEN}✓ Results downloaded${NC}"
echo ""

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo "Test 1 (Evaluation Metrics): $([ $TEST1_RESULT -eq 0 ] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo "Test 2 (EXEC Tracking): $([ $TEST2_RESULT -eq 0 ] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo "Test 3 (Prediction Signals): $([ $TEST3_RESULT -eq 0 ] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo "Test 4 (Unified Architecture): $([ $TEST4_RESULT -eq 0 ] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo ""
if [ $TEST1_RESULT -eq 0 ] && [ $TEST2_RESULT -eq 0 ] && [ $TEST3_RESULT -eq 0 ] && [ $TEST4_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi

