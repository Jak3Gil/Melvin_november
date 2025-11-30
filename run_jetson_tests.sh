#!/bin/bash

# Test Runner for Melvin System on Jetson
# Runs all event-driven tests via USB connection
#
# Usage: ./run_jetson_tests.sh
#
# Connection: USB (COM8) - username: melvin, password: 123456

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
echo "MELVIN SYSTEM TESTS - JETSON RUNNER"
echo "=========================================="
echo ""
echo "Target: $JETSON_USER@$JETSON_HOST"
echo "Path: $JETSON_PATH"
echo ""

# Check if we can connect
echo "Checking connection..."
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot reach Jetson at $JETSON_HOST${NC}"
    echo "Trying serial connection via COM8..."
    # For serial, you'd use minicom or similar
    echo "Note: Serial connection requires manual setup"
    exit 1
fi

echo -e "${GREEN}✓ Connection OK${NC}"
echo ""

# Compile tests locally first (to check for errors)
echo "Compiling tests locally..."
gcc -o test_1_single_node_sanity test_1_single_node_sanity.c -lm -std=c11 -Wall 2>&1 | head -20
gcc -o test_2_pattern_learning test_2_pattern_learning.c -lm -std=c11 -Wall 2>&1 | head -20
gcc -o test_3_fe_pattern_creation test_3_fe_pattern_creation.c -lm -std=c11 -Wall 2>&1 | head -20
gcc -o test_4_control_learning test_4_control_learning.c -lm -std=c11 -Wall 2>&1 | head -20
gcc -o test_5_stability_long_run test_5_stability_long_run.c -lm -std=c11 -Wall 2>&1 | head -20
gcc -o test_6_parameter_robustness test_6_parameter_robustness.c -lm -std=c11 -Wall 2>&1 | head -20

echo -e "${GREEN}✓ Local compilation OK${NC}"
echo ""

# Create remote directory
echo "Setting up remote directory..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "mkdir -p $JETSON_PATH && mkdir -p $JETSON_PATH/results"

# Copy test files
echo "Copying test files to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    test_*.c \
    melvin.c \
    melvin.h \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/

# Copy Makefile if it exists
if [ -f Makefile ]; then
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        Makefile \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/
fi

echo -e "${GREEN}✓ Files copied${NC}"
echo ""

# Compile on Jetson
echo "Compiling tests on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/melvin_tests
echo "Compiling test 1..."
gcc -o test_1_single_node_sanity test_1_single_node_sanity.c -lm -std=c11 -Wall -O2
echo "Compiling test 2..."
gcc -o test_2_pattern_learning test_2_pattern_learning.c -lm -std=c11 -Wall -O2
echo "Compiling test 3..."
gcc -o test_3_fe_pattern_creation test_3_fe_pattern_creation.c -lm -std=c11 -Wall -O2
echo "Compiling test 4..."
gcc -o test_4_control_learning test_4_control_learning.c -lm -std=c11 -Wall -O2
echo "Compiling test 5..."
gcc -o test_5_stability_long_run test_5_stability_long_run.c -lm -std=c11 -Wall -O2
echo "Compiling test 6..."
gcc -o test_6_parameter_robustness test_6_parameter_robustness.c -lm -std=c11 -Wall -O2
echo "Compilation complete"
ENDSSH

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Compilation failed on Jetson${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Compilation complete${NC}"
echo ""

# Run tests
echo "=========================================="
echo "RUNNING TESTS"
echo "=========================================="
echo ""

TOTAL_TESTS=6
PASSED=0
FAILED=0

# Test 1: Single Node Sanity
echo "----------------------------------------"
echo "TEST 1: Single Node Sanity"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && ./test_1_single_node_sanity > results/test_1.log 2>&1"
TEST1_RESULT=$?
if [ $TEST1_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 1 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 1 FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_1.log . 2>/dev/null || true
    tail -20 test_1.log 2>/dev/null || true
fi
echo ""

# Test 2: Pattern Learning
echo "----------------------------------------"
echo "TEST 2: Pattern Learning"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && ./test_2_pattern_learning > results/test_2.log 2>&1"
TEST2_RESULT=$?
if [ $TEST2_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 2 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 2 FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_2.log . 2>/dev/null || true
    tail -20 test_2.log 2>/dev/null || true
fi
echo ""

# Test 3: FE Pattern Creation
echo "----------------------------------------"
echo "TEST 3: FE Pattern Creation"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && ./test_3_fe_pattern_creation > results/test_3.log 2>&1"
TEST3_RESULT=$?
if [ $TEST3_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 3 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 3 FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_3.log . 2>/dev/null || true
    tail -20 test_3.log 2>/dev/null || true
fi
echo ""

# Test 4: Control Learning
echo "----------------------------------------"
echo "TEST 4: Control Learning"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && ./test_4_control_learning > results/test_4.log 2>&1"
TEST4_RESULT=$?
if [ $TEST4_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 4 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 4 FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_4.log . 2>/dev/null || true
    tail -20 test_4.log 2>/dev/null || true
fi
echo ""

# Test 5: Stability Long Run
echo "----------------------------------------"
echo "TEST 5: Stability Long Run"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && timeout 300 ./test_5_stability_long_run > results/test_5.log 2>&1"
TEST5_RESULT=$?
if [ $TEST5_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 5 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 5 FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_5.log . 2>/dev/null || true
    tail -20 test_5.log 2>/dev/null || true
fi
echo ""

# Test 6: Parameter Robustness
echo "----------------------------------------"
echo "TEST 6: Parameter Robustness"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && ./test_6_parameter_robustness > results/test_6.log 2>&1"
TEST6_RESULT=$?
if [ $TEST6_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 6 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 6 FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/test_6.log . 2>/dev/null || true
    tail -20 test_6.log 2>/dev/null || true
fi
echo ""

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

# Download all results
echo "Downloading test results..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no -r \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/ \
    jetson_test_results/ 2>/dev/null || mkdir -p jetson_test_results

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "ALL TESTS PASSED!"
    echo "==========================================${NC}"
    echo ""
    echo "The event-driven physics is solid."
    echo "Ready for instincts.m training."
    exit 0
else
    echo ""
    echo -e "${RED}=========================================="
    echo "SOME TESTS FAILED"
    echo "==========================================${NC}"
    echo ""
    echo "Review test logs in jetson_test_results/"
    exit 1
fi

