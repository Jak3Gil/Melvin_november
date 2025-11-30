#!/bin/bash

# Diagnostic Runner for Melvin System on Jetson
# Runs all diagnostic experiments via SSH

set -e

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="169.254.123.100"
JETSON_PATH="/home/melvin/diagnostics"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "MELVIN DIAGNOSTICS - JETSON RUNNER"
echo "=========================================="
echo ""
echo "Target: $JETSON_USER@$JETSON_HOST"
echo "Path: $JETSON_PATH"
echo ""

# Check connection
echo "Checking connection..."
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot reach Jetson at $JETSON_HOST${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Connection OK${NC}"
echo ""

# Compile locally first (to check for errors)
echo "Compiling diagnostics locally..."
gcc -o diag_experiment_a_single_node diag_experiment_a_single_node.c -lm -std=c11 -Wall -O2 2>&1 | head -20
gcc -o diag_experiment_b_pattern_learning diag_experiment_b_pattern_learning.c -lm -std=c11 -Wall -O2 2>&1 | head -20
gcc -o diag_experiment_c_control_loop diag_experiment_c_control_loop.c -lm -std=c11 -Wall -O2 2>&1 | head -20

echo -e "${GREEN}✓ Local compilation OK${NC}"
echo ""

# Create remote directory
echo "Setting up remote directory..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "mkdir -p $JETSON_PATH && mkdir -p $JETSON_PATH/results"

# Copy diagnostic files
echo "Copying diagnostic files to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    diag_experiment_*.c \
    melvin_diagnostics.h \
    melvin_diagnostics.c \
    melvin.c \
    melvin.h \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/

echo -e "${GREEN}✓ Files copied${NC}"
echo ""

# Compile on Jetson
echo "Compiling diagnostics on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/diagnostics
echo "Compiling Experiment A..."
gcc -o diag_experiment_a_single_node diag_experiment_a_single_node.c -lm -std=c11 -Wall -O2
echo "Compiling Experiment B..."
gcc -o diag_experiment_b_pattern_learning diag_experiment_b_pattern_learning.c -lm -std=c11 -Wall -O2
echo "Compiling Experiment C..."
gcc -o diag_experiment_c_control_loop diag_experiment_c_control_loop.c -lm -std=c11 -Wall -O2
echo "Compilation complete"
ENDSSH

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Compilation failed on Jetson${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Compilation complete${NC}"
echo ""

# Run diagnostics
echo "=========================================="
echo "RUNNING DIAGNOSTICS"
echo "=========================================="
echo ""

TOTAL_EXPERIMENTS=3
PASSED=0
FAILED=0

# Experiment A: Single-Node Sanity
echo "----------------------------------------"
echo "EXPERIMENT A: Single-Node Sanity"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && timeout 120 ./diag_experiment_a_single_node > results/experiment_a.log 2>&1"
EXPERIMENT_A_RESULT=$?
if [ $EXPERIMENT_A_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ EXPERIMENT A PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ EXPERIMENT A FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/experiment_a.log . 2>/dev/null || true
    tail -30 experiment_a.log 2>/dev/null || true
fi
echo ""

# Experiment B: Pattern Learning
echo "----------------------------------------"
echo "EXPERIMENT B: Pattern Learning"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && timeout 180 ./diag_experiment_b_pattern_learning > results/experiment_b.log 2>&1"
EXPERIMENT_B_RESULT=$?
if [ $EXPERIMENT_B_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ EXPERIMENT B PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ EXPERIMENT B FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/experiment_b.log . 2>/dev/null || true
    tail -30 experiment_b.log 2>/dev/null || true
fi
echo ""

# Experiment C: Control Loop
echo "----------------------------------------"
echo "EXPERIMENT C: Control Loop Check"
echo "----------------------------------------"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && timeout 180 ./diag_experiment_c_control_loop > results/experiment_c.log 2>&1"
EXPERIMENT_C_RESULT=$?
if [ $EXPERIMENT_C_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ EXPERIMENT C PASSED${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ EXPERIMENT C FAILED${NC}"
    FAILED=$((FAILED + 1))
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results/experiment_c.log . 2>/dev/null || true
    tail -30 experiment_c.log 2>/dev/null || true
fi
echo ""

# Download all diagnostic results
echo "Downloading diagnostic results..."
mkdir -p jetson_diagnostics_results
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no -r \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/diag_*_results \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/results \
    jetson_diagnostics_results/ 2>/dev/null || true

# Summary
echo "=========================================="
echo "DIAGNOSTICS SUMMARY"
echo "=========================================="
echo ""
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""
echo "Diagnostic CSV files downloaded to: jetson_diagnostics_results/"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "ALL DIAGNOSTICS PASSED!"
    echo "==========================================${NC}"
    echo ""
    echo "Physics verification complete."
    echo "Ready for instincts.m training."
    exit 0
else
    echo -e "${RED}=========================================="
    echo "SOME DIAGNOSTICS FAILED"
    echo "==========================================${NC}"
    echo ""
    echo "Review diagnostic logs and CSV files in jetson_diagnostics_results/"
    echo "Check DIAGNOSTICS_README.md for interpretation guide"
    exit 1
fi

