#!/bin/bash

# Instincts Training Runner for Jetson
# Runs the phased training to build instincts.m v0.1

set -e

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="169.254.123.100"
JETSON_PATH="/home/melvin/instincts_training"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "INSTINCTS TRAINING - JETSON RUNNER"
echo "=========================================="
echo ""
echo "Target: $JETSON_USER@$JETSON_HOST"
echo "Path: $JETSON_PATH"
echo ""

# Check connection
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Jetson"
    exit 1
fi

echo -e "${GREEN}✓ Connection OK${NC}"
echo ""

# Create remote directory
echo "Setting up remote directory..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "mkdir -p $JETSON_PATH && mkdir -p $JETSON_PATH/checkpoints"

# Copy training files
echo "Copying training files..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    instincts_training.c \
    melvin_diagnostics.h \
    melvin_diagnostics.c \
    melvin.c \
    melvin.h \
    $JETSON_USER@$JETSON_HOST:$JETSON_PATH/

echo -e "${GREEN}✓ Files copied${NC}"
echo ""

# Compile on Jetson
echo "Compiling instincts training..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST << 'ENDSSH'
cd /home/melvin/instincts_training
gcc -o instincts_training instincts_training.c -lm -std=c11 -Wall -O2
echo "Compilation complete"
ENDSSH

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo -e "${GREEN}✓ Compilation complete${NC}"
echo ""

# Run training (long-running, recommend screen/tmux)
echo "=========================================="
echo "STARTING INSTINCTS TRAINING"
echo "=========================================="
echo ""
echo -e "${YELLOW}NOTE: This is a long-running process${NC}"
echo "Consider running in screen or tmux:"
echo "  screen -S instincts"
echo "  ./instincts_training"
echo ""
echo "Or run directly:"
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST \
    "cd $JETSON_PATH && nohup ./instincts_training > training.log 2>&1 &"

echo "Training started in background"
echo "Check progress: ssh melvin@169.254.123.100 'tail -f /home/melvin/instincts_training/training.log'"
echo ""
echo "Checkpoints will be saved to: $JETSON_PATH/checkpoints/"
echo "Metrics logged to: $JETSON_PATH/training_metrics.csv"

