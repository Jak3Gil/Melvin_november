#!/bin/bash
# deploy_to_jetson.sh - Deploy Melvin code from Mac to Jetson
# Usage: ./deploy_to_jetson.sh [reset_brain]

set -e

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_MELVIN_DIR="~/melvin"
JETSON_BRAIN_PATH="/mnt/melvin_ssd/melvin_brain/brain.m"

RESET_BRAIN=false
if [ "$1" == "reset_brain" ] || [ "$1" == "--reset-brain" ]; then
    RESET_BRAIN=true
fi

echo "=========================================="
echo "Melvin Deployment to Jetson"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "src/melvin.c" ]; then
    echo "Error: Must run from Melvin_november directory"
    exit 1
fi

echo "1. Stopping Melvin on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "killall -9 melvin_hardware_runner 2>/dev/null; sleep 2; echo '  ✓ Stopped'"

echo ""
echo "2. Backing up current brain.m (if exists)..."
if sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "[ -f $JETSON_BRAIN_PATH ]"; then
    BACKUP_NAME="brain.m.backup.$(date +%Y%m%d_%H%M%S)"
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
        "cp $JETSON_BRAIN_PATH /mnt/melvin_ssd/melvin_brain/$BACKUP_NAME && echo '  ✓ Backed up to $BACKUP_NAME'"
fi

if [ "$RESET_BRAIN" = true ]; then
    echo ""
    echo "3. Resetting brain.m (fresh start)..."
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
        "rm -f $JETSON_BRAIN_PATH && echo '  ✓ brain.m removed (will be recreated)'"
else
    echo ""
    echo "3. Preserving brain.m (keeping learned patterns)..."
    echo "  ✓ brain.m will be preserved"
    echo "  (Use './deploy_to_jetson.sh reset_brain' to start fresh)"
fi

echo ""
echo "4. Copying source files to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    src/melvin.c \
    src/melvin.h \
    src/host_syscalls.c \
    src/melvin_tools.c \
    src/melvin_tools.h \
    src/melvin_tool_layer.c \
    src/melvin_hardware_audio.c \
    src/melvin_hardware_video.c \
    src/melvin_hardware_runner.c \
    src/melvin_hardware.h \
    "$JETSON_USER@$JETSON_IP:$JETSON_MELVIN_DIR/src/" 2>&1 | tail -1
echo "  ✓ Source files copied"

echo ""
echo "5. Rebuilding on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'REMOTE_BUILD'
cd ~/melvin

echo "  Compiling..."
gcc -std=c11 -Wall -O2 -o melvin_hardware_runner \
    src/melvin_hardware_runner.c \
    src/melvin.c \
    src/host_syscalls.c \
    src/melvin_tools.c \
    src/melvin_tool_layer.c \
    src/melvin_hardware_audio.c \
    src/melvin_hardware_video.c \
    -lm -pthread -lasound -lv4l2 2>&1 | grep -E "(error|fatal|undefined)" | head -5 || echo "  ✓ Compiled successfully"

if [ -f melvin_hardware_runner ]; then
    echo "  ✓ Binary built"
else
    echo "  ✗ Build failed"
    exit 1
fi
REMOTE_BUILD

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Build failed on Jetson"
    exit 1
fi

echo ""
echo "6. Starting Melvin on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'REMOTE_START'
cd ~/melvin

nohup ./melvin_hardware_runner /mnt/melvin_ssd/melvin_brain/brain.m default default > /mnt/melvin_ssd/melvin_brain/melvin.log 2>&1 &
NEW_PID=$!

echo "  ✓ Started: PID $NEW_PID"
sleep 5

if ps aux | grep melvin_hardware_runner | grep -v grep > /dev/null; then
    echo "  ✓ Melvin is running"
else
    echo "  ✗ Failed to start - check logs:"
    tail -20 /mnt/melvin_ssd/melvin_brain/melvin.log
    exit 1
fi
REMOTE_START

echo ""
echo "=========================================="
echo "✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Brain location: $JETSON_BRAIN_PATH"
if [ "$RESET_BRAIN" = true ]; then
    echo "Brain status: Fresh (reset)"
else
    echo "Brain status: Preserved (learned patterns kept)"
fi
echo ""
echo "Monitor: ssh $JETSON_USER@$JETSON_IP 'tail -f ~/melvin/melvin.log'"
echo "Stop: ssh $JETSON_USER@$JETSON_IP 'killall melvin_hardware_runner'"

