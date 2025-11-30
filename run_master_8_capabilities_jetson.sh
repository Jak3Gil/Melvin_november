#!/bin/bash

# Test Runner for test_master_8_capabilities on Jetson
# This is the MASTER TEST SUITE that verifies all 8 core capabilities
#
# Usage: ./run_master_8_capabilities_jetson.sh
#
# Connection: Direct ethernet at 169.254.123.100 (or USB/COM8 via SSH tunnel)

set -e

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="169.254.123.100"  # Direct ethernet, or use COM8 for serial
JETSON_PATH="/home/melvin/melvin_tests"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "MASTER TEST SUITE — 8 Core Capabilities"
echo "JETSON RUNNER"
echo "=========================================="
echo ""
echo "Target: $JETSON_USER@$JETSON_HOST"
echo "Path: $JETSON_PATH"
echo ""
echo "This test suite answers:"
echo "\"Is Melvin.m behaving like a real, stable, executable brain?\""
echo ""
echo "The 8 capabilities tested:"
echo "1. INPUT → GRAPH → OUTPUT (No Cheating)"
echo "2. Graph-Driven Execution (No Direct C Calls)"
echo "3. Stability + Safety Under Stress"
echo "4. Correctness of Basic Tools (ADD, MUL, etc.)"
echo "5. Multi-Hop Reasoning (Chain of Tools)"
echo "6. Tool Selection (Branching Behavior)"
echo "7. Learning Tests (Co-Activity, Error Reduction)"
echo "8. Long-Run Stability (No Drift, No Corruption)"
echo ""

# Check if we can connect
echo "Checking connection..."
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo -e "${YELLOW}WARNING: Cannot reach Jetson at $JETSON_HOST${NC}"
    echo "If using USB/COM8 serial, ensure SSH tunnel is set up"
    echo "Or check network connection"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ Connection OK${NC}"
echo ""

# Check for required files
echo "Checking required files..."
REQUIRED_FILES=(
    "test_master_8_capabilities.c"
    "melvin.c"
    "instincts.c"
    "test_helpers.h"
    "melvin_instincts.h"
    "melvin_exec_helpers.c"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo -e "${RED}ERROR: Missing required files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo -e "${GREEN}✓ All required files present${NC}"
echo ""

# Compile test locally first (to check for errors)
echo "Compiling test locally (syntax check)..."
if ! gcc -o test_master_8_capabilities test_master_8_capabilities.c -lm -std=c11 -Wall -O0 2>&1 | head -50; then
    echo -e "${RED}ERROR: Local compilation failed${NC}"
    echo "Please fix compilation errors before running on Jetson"
    exit 1
fi

echo -e "${GREEN}✓ Local compilation OK${NC}"
echo ""

# Create remote directory
echo "Setting up remote directory..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "mkdir -p $JETSON_PATH && mkdir -p $JETSON_PATH/results" 2>/dev/null || {
    echo -e "${YELLOW}WARNING: Could not create remote directory via SSH${NC}"
    echo "If using USB/COM8, you may need to set up SSH tunnel manually"
    exit 1
}

# Copy test files
echo "Copying test files to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    test_master_8_capabilities.c \
    melvin.c \
    instincts.c \
    test_helpers.h \
    melvin_instincts.h \
    melvin_exec_helpers.c \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/ 2>/dev/null || {
    echo -e "${RED}ERROR: Failed to copy files to Jetson${NC}"
    exit 1
}

echo -e "${GREEN}✓ Files copied${NC}"
echo ""

# Compile on Jetson
echo "Compiling test on Jetson..."
echo "This may take a minute..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/melvin_tests
echo "Compiling test_master_8_capabilities..."
gcc -o test_master_8_capabilities test_master_8_capabilities.c -lm -std=c11 -Wall -O0 2>&1 | head -50
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
echo "RUNNING MASTER TEST SUITE"
echo "=========================================="
echo ""
echo "This will test all 8 capabilities..."
echo "It may take several minutes to complete."
echo ""

# Run with timeout (30 minutes max)
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && timeout 1800 ./test_master_8_capabilities > results/test_master_8_capabilities.log 2>&1"
TEST_RESULT=$?

# Download results
echo ""
echo "Downloading test results..."
mkdir -p jetson_test_results
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_master_8_capabilities.log \
    jetson_test_results/ 2>/dev/null || true

# Show results
echo ""
echo "=========================================="
echo "TEST RESULTS"
echo "=========================================="
echo ""

if [ -f jetson_test_results/test_master_8_capabilities.log ]; then
    cat jetson_test_results/test_master_8_capabilities.log
else
    echo "Could not download test log"
    echo "Test exit code: $TEST_RESULT"
fi

echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ ALL 8 CAPABILITIES VERIFIED${NC}"
    echo ""
    echo "SUCCESS: Melvin.m is a self-executing computational substrate"
    echo "capable of building intelligence on top."
    echo ""
    echo "All capabilities passed:"
    echo "  1. INPUT → GRAPH → OUTPUT (No Cheating) ✓"
    echo "  2. Graph-Driven Execution (No Direct C Calls) ✓"
    echo "  3. Stability + Safety Under Stress ✓"
    echo "  4. Correctness of Basic Tools (ADD, MUL, etc.) ✓"
    echo "  5. Multi-Hop Reasoning (Chain of Tools) ✓"
    echo "  6. Tool Selection (Branching Behavior) ✓"
    echo "  7. Learning Tests (Co-Activity, Error Reduction) ✓"
    echo "  8. Long-Run Stability (No Drift, No Corruption) ✓"
    echo ""
    exit 0
else
    echo -e "${RED}✗ SOME CAPABILITIES FAILED${NC}"
    echo ""
    echo "If any capability fails:"
    echo "- Intelligence cannot form"
    echo "- Reasoning will collapse"
    echo "- Patterns will not stabilize"
    echo "- EXEC cannot be trusted"
    echo "- Learning cannot scale"
    echo "- Persistence will break"
    echo ""
    echo "Review test log: jetson_test_results/test_master_8_capabilities.log"
    echo ""
    exit 1
fi

