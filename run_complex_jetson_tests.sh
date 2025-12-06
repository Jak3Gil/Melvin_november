#!/bin/bash
# run_complex_jetson_tests.sh - Run comprehensive complex tests on Jetson
# 
# Tests:
# 1. Evaluation Metrics (5-domain test)
# 2. Master 8 Capabilities
# 3. Scaling Proof
# 4. Comprehensive Routing
# 5. EXEC Learning
# 6. Multi-hop Patterns
# 7. Meta Learning
# 8. EXEC Tracking (new feature)

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
NC='\033[0m' # No Color

echo "=========================================="
echo "COMPLEX JETSON TEST SUITE"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Target: $JETSON_USER@$JETSON_IP"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check connection
echo "Checking connection to Jetson..."
if ! ping -c 1 -W 2 $JETSON_IP > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot reach Jetson at $JETSON_IP${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Connection OK${NC}"
echo ""

# Transfer source files
echo "Transferring source files to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    src/melvin.c src/melvin.h \
    "$JETSON_USER@$JETSON_IP:$WORK_DIR/" || {
    echo -e "${RED}ERROR: Failed to transfer source files${NC}"
    exit 1
}
# Also copy to root as melvin.c (some tests expect it there)
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "cd $WORK_DIR && cp src/melvin.c melvin.c && cp src/melvin.h melvin.h" 2>/dev/null || true
echo -e "${GREEN}✓ Source files transferred${NC}"
echo ""

# Transfer test files
echo "Transferring test files to Jetson..."
TEST_FILES=(
    "evaluate_melvin_metrics.c"
    "test/test_master_8_capabilities.c"
    "test/test_scaling_proof.c"
    "test/test_comprehensive_routing.c"
    "test/test_exec_learning.c"
    "test/test_multi_hop_patterns.c"
    "test/test_meta_learning.c"
    "test_exec_tracking.c"
)

for test_file in "${TEST_FILES[@]}"; do
    if [ -f "$test_file" ]; then
        sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
            "$test_file" \
            "$JETSON_USER@$JETSON_IP:$WORK_DIR/" || {
            echo -e "${YELLOW}⚠ Warning: Failed to transfer $test_file${NC}"
        }
    else
        echo -e "${YELLOW}⚠ Warning: $test_file not found, skipping${NC}"
    fi
done
echo -e "${GREEN}✓ Test files transferred${NC}"
echo ""

# Run tests on Jetson
echo "=========================================="
echo "RUNNING TESTS ON JETSON"
echo "=========================================="
echo ""

TOTAL_TESTS=0
PASSED=0
FAILED=0

# Test 1: Evaluation Metrics (5-domain test)
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}TEST 1: Evaluation Metrics (5-domain)${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
    echo "Compiling evaluate_melvin_metrics..."
    # Check if melvin_run_physics exists in source (check both locations)
    if grep -q "void melvin_run_physics" melvin.c 2>/dev/null || grep -q "void melvin_run_physics" src/melvin.c 2>/dev/null; then
        echo "✓ melvin_run_physics found in source"
    else
        echo "✗ melvin_run_physics NOT found in source!"
    fi
    # Try melvin.c first (root), fallback to src/melvin.c
    if [ -f melvin.c ]; then
        gcc -o evaluate_melvin_metrics evaluate_melvin_metrics.c melvin.c -I. -std=c11 -lm -O2 2>&1 | head -40
    else
        gcc -o evaluate_melvin_metrics evaluate_melvin_metrics.c src/melvin.c -Isrc -I. -std=c11 -lm -O2 2>&1 | head -40
    fi
if [ $? -eq 0 ]; then
    echo "Running evaluation metrics test..."
    ./evaluate_melvin_metrics > evaluation_metrics_$(date +%Y%m%d_%H%M%S).log 2>&1
    EXIT_CODE=$?
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
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 1 FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 2: Master 8 Capabilities
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}TEST 2: Master 8 Capabilities${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
echo "Compiling test_master_8_capabilities..."
gcc -o test_master_8_capabilities test/test_master_8_capabilities.c src/melvin.c -I. -std=c11 -lm -O2 2>&1 | head -20
if [ $? -eq 0 ]; then
    echo "Running master 8 capabilities test..."
    ./test_master_8_capabilities > master_8_capabilities_$(date +%Y%m%d_%H%M%S).log 2>&1
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST2_RESULT=$?
if [ $TEST2_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 2 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 2 FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 3: Scaling Proof
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}TEST 3: Scaling Proof${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
echo "Compiling test_scaling_proof..."
gcc -o test_scaling_proof test/test_scaling_proof.c src/melvin.c -I. -std=c11 -lm -O2 2>&1 | head -20
if [ $? -eq 0 ]; then
    echo "Running scaling proof test..."
    ./test_scaling_proof > scaling_proof_$(date +%Y%m%d_%H%M%S).log 2>&1
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
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 3 FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 4: Comprehensive Routing
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}TEST 4: Comprehensive Routing${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
echo "Compiling test_comprehensive_routing..."
gcc -o test_comprehensive_routing test/test_comprehensive_routing.c src/melvin.c -I. -std=c11 -lm -O2 2>&1 | head -20
if [ $? -eq 0 ]; then
    echo "Running comprehensive routing test..."
    ./test_comprehensive_routing > comprehensive_routing_$(date +%Y%m%d_%H%M%S).log 2>&1
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST4_RESULT=$?
if [ $TEST4_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 4 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 4 FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 5: EXEC Learning
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}TEST 5: EXEC Learning${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
echo "Compiling test_exec_learning..."
gcc -o test_exec_learning test/test_exec_learning.c src/melvin.c -I. -std=c11 -lm -O2 2>&1 | head -20
if [ $? -eq 0 ]; then
    echo "Running EXEC learning test..."
    ./test_exec_learning > exec_learning_$(date +%Y%m%d_%H%M%S).log 2>&1
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST5_RESULT=$?
if [ $TEST5_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 5 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 5 FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 6: Multi-hop Patterns
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}TEST 6: Multi-hop Patterns${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
echo "Compiling test_multi_hop_patterns..."
gcc -o test_multi_hop_patterns test/test_multi_hop_patterns.c src/melvin.c -I. -std=c11 -lm -O2 2>&1 | head -20
if [ $? -eq 0 ]; then
    echo "Running multi-hop patterns test..."
    ./test_multi_hop_patterns > multi_hop_patterns_$(date +%Y%m%d_%H%M%S).log 2>&1
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST6_RESULT=$?
if [ $TEST6_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 6 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 6 FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 7: Meta Learning
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}TEST 7: Meta Learning${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
echo "Compiling test_meta_learning..."
gcc -o test_meta_learning test/test_meta_learning.c src/melvin.c -I. -std=c11 -lm -O2 2>&1 | head -20
if [ $? -eq 0 ]; then
    echo "Running meta learning test..."
    ./test_meta_learning > meta_learning_$(date +%Y%m%d_%H%M%S).log 2>&1
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST7_RESULT=$?
if [ $TEST7_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 7 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 7 FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 8: EXEC Tracking (new feature)
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}TEST 8: EXEC Tracking${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
mkdir -p evaluation_results
echo "Compiling test_exec_tracking..."
gcc -o test_exec_tracking test_exec_tracking.c src/melvin.c -I. -std=c11 -lm -O2 2>&1 | head -20
if [ $? -eq 0 ]; then
    echo "Running EXEC tracking test..."
    ./test_exec_tracking > exec_tracking_$(date +%Y%m%d_%H%M%S).log 2>&1
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE"
    # Check if log file was created
    if [ -f evaluation_results/exec_creation.log ]; then
        echo "✓ exec_creation.log created"
        wc -l evaluation_results/exec_creation.log
    else
        echo "⚠ exec_creation.log not found"
    fi
    exit $EXIT_CODE
else
    echo "Compilation failed"
    exit 1
fi
ENDSSH
TEST8_RESULT=$?
if [ $TEST8_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ TEST 8 PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ TEST 8 FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Collect results
echo "=========================================="
echo "COLLECTING RESULTS"
echo "=========================================="
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'ENDSSH'
cd /home/melvin/melvin
echo "Collecting log files..."
ls -lh *.log evaluation_results/*.log evaluation_results/*.csv 2>/dev/null | head -20
ENDSSH

# Download results
echo "Downloading results..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    "$JETSON_USER@$JETSON_IP:$WORK_DIR/*.log" \
    "$RESULTS_DIR/" 2>/dev/null || true

sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    "$JETSON_USER@$JETSON_IP:$WORK_DIR/evaluation_results/*" \
    "$RESULTS_DIR/" 2>/dev/null || true

echo -e "${GREEN}✓ Results downloaded${NC}"
echo ""

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi

