#!/bin/bash

# Test Runner for EXEC Smoke Test and Master 8 Capabilities on Jetson via USB
# This runs both the minimal EXEC smoke test and the full master test suite
#
# Usage: ./run_tests_jetson_usb.sh

set -e

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="169.254.123.100"  # Direct ethernet (or USB/COM8 via SSH tunnel)
JETSON_PATH="/home/melvin/melvin_tests"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "MELVIN TESTS — JETSON RUNNER (USB/Ethernet)"
echo "=========================================="
echo ""
echo "Target: $JETSON_USER@$JETSON_HOST"
echo "Path: $JETSON_PATH"
echo ""
echo "Tests to run:"
echo "  1. test_0_0_exec_smoke (Minimal EXEC smoke test)"
echo "  2. test_master_8_capabilities (Full 8-capability suite)"
echo ""

# Check if we can connect
echo "Checking connection..."
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo -e "${YELLOW}WARNING: Cannot reach Jetson at $JETSON_HOST${NC}"
    echo "If using USB/COM8 serial, ensure SSH tunnel is set up"
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
    "test_0_0_exec_smoke.c"
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

# Create remote directory
echo "Setting up remote directory..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "mkdir -p $JETSON_PATH && mkdir -p $JETSON_PATH/results" 2>/dev/null || {
    echo -e "${YELLOW}WARNING: Could not create remote directory${NC}"
    exit 1
}

# Copy test files
echo "Copying test files to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    test_0_0_exec_smoke.c \
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

# Compile both tests on Jetson
echo "=========================================="
echo "COMPILING TESTS ON JETSON"
echo "=========================================="
echo ""

echo -e "${BLUE}Compiling test_0_0_exec_smoke...${NC}"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/melvin_tests
echo "Compiling test_0_0_exec_smoke..."
gcc -o test_0_0_exec_smoke test_0_0_exec_smoke.c -lm -std=c11 -Wall -O0 2>&1 | head -30
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi
echo "✓ test_0_0_exec_smoke compiled"
ENDSSH

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Smoke test compilation failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Compiling test_master_8_capabilities...${NC}"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/melvin_tests
echo "Compiling test_master_8_capabilities..."
gcc -o test_master_8_capabilities test_master_8_capabilities.c -lm -std=c11 -Wall -O0 2>&1 | head -30
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi
echo "✓ test_master_8_capabilities compiled"
ENDSSH

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Master test compilation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All tests compiled${NC}"
echo ""

# Run smoke test first
echo "=========================================="
echo "TEST 1: EXEC SMOKE TEST"
echo "=========================================="
echo ""
echo "Running minimal EXEC smoke test (2 + 3 = 5)..."
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && timeout 60 ./test_0_0_exec_smoke > results/test_0_0_exec_smoke.log 2>&1"
SMOKE_RESULT=$?

# Download smoke test results
mkdir -p jetson_test_results
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_0_0_exec_smoke.log \
    jetson_test_results/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "SMOKE TEST OUTPUT"
echo "=========================================="
echo ""

if [ -f jetson_test_results/test_0_0_exec_smoke.log ]; then
    cat jetson_test_results/test_0_0_exec_smoke.log
else
    echo "Could not download smoke test log"
    echo "Test exit code: $SMOKE_RESULT"
fi

echo ""
if [ $SMOKE_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ SMOKE TEST PASSED${NC}"
else
    echo -e "${RED}✗ SMOKE TEST FAILED${NC}"
fi
echo ""

# Run master test suite
echo "=========================================="
echo "TEST 2: MASTER 8 CAPABILITIES TEST"
echo "=========================================="
echo ""
echo "Running full master test suite..."
echo "This will test all 8 capabilities and may take several minutes."
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && timeout 1800 ./test_master_8_capabilities > results/test_master_8_capabilities.log 2>&1"
MASTER_RESULT=$?

# Download master test results
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_master_8_capabilities.log \
    jetson_test_results/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "MASTER TEST OUTPUT"
echo "=========================================="
echo ""

if [ -f jetson_test_results/test_master_8_capabilities.log ]; then
    cat jetson_test_results/test_master_8_capabilities.log
else
    echo "Could not download master test log"
    echo "Test exit code: $MASTER_RESULT"
fi

echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo ""

if [ $SMOKE_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ EXEC Smoke Test: PASSED${NC}"
else
    echo -e "${RED}✗ EXEC Smoke Test: FAILED${NC}"
fi

if [ $MASTER_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Master 8 Capabilities: PASSED${NC}"
else
    echo -e "${RED}✗ Master 8 Capabilities: FAILED${NC}"
fi

echo ""
echo "Full logs saved in: jetson_test_results/"
echo "  - test_0_0_exec_smoke.log"
echo "  - test_master_8_capabilities.log"
echo ""

if [ $SMOKE_RESULT -eq 0 ] && [ $MASTER_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi

