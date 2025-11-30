#!/bin/bash

# Test inject_full_instincts on Jetson via USB/ethernet
# Tests the fixed typedef conflict resolution

set -e

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="169.254.123.100"  # Direct ethernet, or use COM8 for serial
JETSON_PATH="/home/melvin/melvin_tests"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "TEST INJECT_FULL_INSTINCTS ON JETSON"
echo "=========================================="
echo ""
echo "Target: $JETSON_USER@$JETSON_HOST"
echo "Path: $JETSON_PATH"
echo ""

# Check connection
echo "Checking connection..."
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot reach Jetson at $JETSON_HOST${NC}"
    echo "Note: If using COM8 serial, connection requires manual setup"
    exit 1
fi

echo -e "${GREEN}✓ Connection OK${NC}"
echo ""

# Compile locally first (to check for errors)
echo "Compiling inject_full_instincts locally..."
gcc -std=c11 -O0 -Wall -Wextra inject_full_instincts.c -lm -o inject_full_instincts 2>&1 | grep -i "error\|undefined\|conflict" | head -20 || true

if [ ! -f inject_full_instincts ]; then
    echo -e "${RED}ERROR: Local compilation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Local compilation OK${NC}"
echo ""

# Create remote directory
echo "Setting up remote directory..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "mkdir -p $JETSON_PATH && mkdir -p $JETSON_PATH/results"

# Copy required files
echo "Copying files to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    inject_full_instincts.c \
    instincts.c \
    melvin.c \
    melvin.h \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/

echo -e "${GREEN}✓ Files copied${NC}"
echo ""

# Compile on Jetson
echo "Compiling inject_full_instincts on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/melvin_tests
echo "Compiling inject_full_instincts..."
gcc -std=c11 -O2 -Wall -Wextra inject_full_instincts.c -lm -o inject_full_instincts 2>&1 | head -30
if [ -f inject_full_instincts ]; then
    echo "✓ Compilation successful"
else
    echo "✗ Compilation failed"
    exit 1
fi
ENDSSH

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Compilation failed on Jetson${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Compilation complete${NC}"
echo ""

# Run test
echo "=========================================="
echo "RUNNING INJECT_FULL_INSTINCTS TEST"
echo "=========================================="
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/melvin_tests
rm -f test_melvin.m
echo "Running inject_full_instincts..."
./inject_full_instincts test_melvin.m 2>&1 | tee results/inject_test.log
INJECT_RESULT=$?
if [ $INJECT_RESULT -eq 0 ]; then
    echo ""
    echo "✓ Injection successful"
    if [ -f test_melvin.m ]; then
        echo "✓ File created: test_melvin.m"
        ls -lh test_melvin.m
    else
        echo "✗ File not created"
        exit 1
    fi
else
    echo ""
    echo "✗ Injection failed (exit code: $INJECT_RESULT)"
    exit 1
fi
ENDSSH

TEST_RESULT=$?

# Download results
echo ""
echo "Downloading test results..."
mkdir -p jetson_test_results
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/inject_test.log \
    jetson_test_results/ 2>/dev/null || true

# Summary
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST PASSED${NC}"
    echo ""
    echo "The typedef conflict fix works correctly on Jetson!"
    echo "Results saved to: jetson_test_results/inject_test.log"
    exit 0
else
    echo -e "${RED}✗ TEST FAILED${NC}"
    echo ""
    echo "Check logs in jetson_test_results/inject_test.log"
    if [ -f jetson_test_results/inject_test.log ]; then
        echo ""
        echo "Last 30 lines of log:"
        tail -30 jetson_test_results/inject_test.log
    fi
    exit 1
fi

