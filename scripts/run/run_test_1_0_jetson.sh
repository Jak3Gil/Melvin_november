#!/bin/bash

# Test Runner for test_1_0_graph_add32 on Jetson
# This is the REAL agent test where graph+EXEC computes a+b
#
# Usage: ./run_test_1_0_jetson.sh
#
# Connection: Direct ethernet at 169.254.123.100

set -e

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="169.254.123.100"
JETSON_PATH="/home/melvin/melvin_tests"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "TEST 1.0 — Real Graph-Driven ADD32"
echo "JETSON RUNNER"
echo "=========================================="
echo ""
echo "Target: $JETSON_USER@$JETSON_HOST"
echo "Path: $JETSON_PATH"
echo ""
echo "This test proves Melvin can compute end-to-end:"
echo "  - Harness only: sets inputs, ticks, checks outputs"
echo "  - Graph + EXEC: performs ALL computation (read, add, write)"
echo "  - Harness NEVER computes a+b (except for ground truth)"
echo ""

# Check if we can connect
echo "Checking connection..."
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot reach Jetson at $JETSON_HOST${NC}"
    echo "Please check network connection"
    exit 1
fi

echo -e "${GREEN}✓ Connection OK${NC}"
echo ""

# Compile test locally first (to check for errors)
echo "Compiling test locally (syntax check)..."
if ! gcc -o test_1_0_graph_add32 test_1_0_graph_add32.c -lm -std=c11 -Wall -O0 2>&1 | head -30; then
    echo -e "${RED}ERROR: Local compilation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Local compilation OK${NC}"
echo ""

# Create remote directory
echo "Setting up remote directory..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "mkdir -p $JETSON_PATH && mkdir -p $JETSON_PATH/results"

# Copy test files
echo "Copying test files to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    test_1_0_graph_add32.c \
    melvin.c \
    instincts.c \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/

echo -e "${GREEN}✓ Files copied${NC}"
echo ""

# Compile on Jetson
echo "Compiling test on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/melvin_tests
echo "Compiling test_1_0_graph_add32..."
gcc -o test_1_0_graph_add32 test_1_0_graph_add32.c -lm -std=c11 -Wall -O0 2>&1 | head -30
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi
echo "Compilation complete"
ENDSSH

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Compilation failed on Jetson${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Compilation complete${NC}"
echo ""

# Run test
echo "=========================================="
echo "RUNNING TEST 1.0"
echo "=========================================="
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && ./test_1_0_graph_add32 > results/test_1_0.log 2>&1"
TEST_RESULT=$?

# Download results
echo "Downloading test results..."
mkdir -p jetson_test_results
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_1_0.log \
    jetson_test_results/ 2>/dev/null || true

# Show results
echo ""
echo "=========================================="
echo "TEST RESULTS"
echo "=========================================="
echo ""

if [ -f jetson_test_results/test_1_0.log ]; then
    cat jetson_test_results/test_1_0.log
else
    echo "Could not download test log"
fi

echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 1.0 PASSED${NC}"
    echo ""
    echo "SUCCESS: Melvin's graph + EXEC computed all additions correctly!"
    echo "This proves Melvin can act like a real program, not just labeled memory."
    echo ""
    exit 0
else
    echo -e "${RED}✗ TEST 1.0 FAILED${NC}"
    echo ""
    echo "Review test log: jetson_test_results/test_1_0.log"
    echo ""
    exit 1
fi

