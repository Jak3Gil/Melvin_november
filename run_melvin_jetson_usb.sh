#!/bin/bash
# run_melvin_jetson_usb.sh - Compile and run Melvin on Jetson via USB

set -e

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
WORK_DIR="/mnt/melvin_ssd/melvin"
BRAIN_PATH="$WORK_DIR/brain.m"

echo "=========================================="
echo "Running Melvin on Jetson via USB"
echo "=========================================="
echo "Target: $JETSON_USER@$JETSON_IP"
echo "Work Dir: $WORK_DIR"
echo ""

# Check connection
echo "Checking connection to Jetson..."
if ! ping -c 1 -W 2 $JETSON_IP > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Jetson at $JETSON_IP"
    echo "Make sure USB connection is active and Jetson is accessible"
    exit 1
fi
echo "✓ Connection OK"
echo ""

# Compile melvin_run_continuous locally first (syntax check)
echo "Compiling melvin_run_continuous locally (syntax check)..."
gcc -std=c11 -Wall -Wextra -O2 -c src/melvin.c -o /tmp/melvin.o -lm -pthread 2>&1 | grep -v "unused" || true
gcc -std=c11 -Wall -Wextra -O2 -c src/melvin_run_continuous.c -o /tmp/melvin_run_continuous.o 2>&1 | grep -v "unused" || true
gcc -std=c11 -Wall -Wextra -O2 -c src/host_syscalls.c -o /tmp/host_syscalls.o 2>&1 | grep -v "unused" || true
echo "✓ Local compilation OK"
echo ""

# Transfer source files to Jetson
echo "Transferring source files to Jetson..."
# Use home directory first, then move if needed
TEMP_DIR="~/melvin_build"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "mkdir -p $TEMP_DIR/src 2>/dev/null; mkdir -p $WORK_DIR/src 2>/dev/null || mkdir -p ~/melvin/src" || true
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no src/melvin.c src/melvin.h src/melvin_run_continuous.c src/host_syscalls.c "$JETSON_USER@$JETSON_IP:~/melvin/src/" || {
    echo "ERROR: Failed to transfer source files"
    exit 1
}
echo "✓ Files transferred"
echo ""

# Compile on Jetson
echo "Compiling on Jetson..."
echo "This may take a minute..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
cd ~/melvin

echo "Compiling melvin_run_continuous..."
gcc -std=c11 -Wall -Wextra -O2 -o melvin_run_continuous \
    src/melvin.c src/melvin_run_continuous.c src/host_syscalls.c \
    -lm -pthread 2>&1 | grep -v "unused" || true

if [ ! -f melvin_run_continuous ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

chmod +x melvin_run_continuous

# Try to copy to work directory if it exists and is writable
if [ -w /mnt/melvin_ssd/melvin ] 2>/dev/null; then
    cp melvin_run_continuous /mnt/melvin_ssd/melvin/ 2>/dev/null || true
fi

echo "✓ Compilation complete"
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed on Jetson"
    exit 1
fi
echo ""

# Check if brain file exists, create if needed
echo "Checking for brain file..."
# Also check if we can access it (permission check)
BRAIN_ACCESSIBLE=false
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "test -r $BRAIN_PATH && test -w $BRAIN_PATH" && BRAIN_ACCESSIBLE=true || {
    # Try to copy brain to home directory if not accessible
    echo "Brain file not accessible, copying to home directory..."
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
if [ -f /mnt/melvin_ssd/melvin/brain.m ] && [ ! -f ~/melvin/brain.m ]; then
    sudo cp /mnt/melvin_ssd/melvin/brain.m ~/melvin/brain.m 2>/dev/null || cp /mnt/melvin_ssd/melvin/brain.m ~/melvin/brain.m 2>/dev/null || true
    sudo chown melvin:melvin ~/melvin/brain.m 2>/dev/null || chown melvin:melvin ~/melvin/brain.m 2>/dev/null || true
fi
EOF
    BRAIN_PATH="~/melvin/brain.m"
}

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "test -f $BRAIN_PATH || test -f ~/melvin/brain.m" || {
    echo "Brain file not found. Creating brain..."
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
cd /mnt/melvin_ssd/melvin

# Check if melvin_pack_corpus exists, if not compile it
if [ ! -f melvin_pack_corpus ]; then
    echo "Compiling melvin_pack_corpus..."
    if [ -f src/melvin_pack_corpus.c ]; then
        gcc -std=c11 -Wall -Wextra -O2 -o melvin_pack_corpus \
            src/melvin_pack_corpus.c src/melvin.c -lm -pthread
        chmod +x melvin_pack_corpus
    fi
fi

# Check if melvin_seed_instincts exists, if not compile it
if [ ! -f melvin_seed_instincts ]; then
    echo "Compiling melvin_seed_instincts..."
    if [ -f src/melvin_seed_instincts.c ]; then
        gcc -std=c11 -Wall -Wextra -O2 -o melvin_seed_instincts \
            src/melvin_seed_instincts.c src/melvin.c -lm -pthread
        chmod +x melvin_seed_instincts
    fi
fi

# Create brain if tools exist
if [ -f melvin_pack_corpus ]; then
    if [ -d corpus/basic ]; then
        echo "Packing corpus into brain..."
        ./melvin_pack_corpus -i corpus/basic -o brain.m \
            --hot-nodes 10000 --hot-edges 50000 --hot-blob-bytes 1048576
    else
        echo "Creating empty brain..."
        ./melvin_pack_corpus -i /tmp -o brain.m \
            --hot-nodes 10000 --hot-edges 50000 --hot-blob-bytes 1048576 --cold-data-bytes 0
    fi
    
    if [ -f melvin_seed_instincts ]; then
        echo "Seeding instincts..."
        ./melvin_seed_instincts brain.m
    fi
else
    echo "WARNING: melvin_pack_corpus not available, using existing brain or creating minimal one"
fi
EOF
}

# Check if brain exists now (try both locations)
BRAIN_FOUND=false
if sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "test -f $BRAIN_PATH" 2>/dev/null; then
    BRAIN_FOUND=true
    BRAIN_PATH="/mnt/melvin_ssd/melvin/brain.m"
elif sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "test -f ~/melvin/brain.m" 2>/dev/null; then
    BRAIN_FOUND=true
    BRAIN_PATH="~/melvin/brain.m"
fi

if [ "$BRAIN_FOUND" = false ]; then
    echo "ERROR: Brain file not found"
    echo "Searched: /mnt/melvin_ssd/melvin/brain.m and ~/melvin/brain.m"
    exit 1
fi
echo "✓ Brain file ready at $BRAIN_PATH"
echo ""

# Run melvin
echo "=========================================="
echo "Starting Melvin on Jetson"
echo "=========================================="
echo "Brain: $BRAIN_PATH"
echo "Press Ctrl+C to stop"
echo ""

# Run melvin - use home directory brain (accessible) and executable
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
cd ~/melvin

# Ensure brain is accessible
if [ ! -f brain.m ] && [ -f /mnt/melvin_ssd/melvin/brain.m ]; then
    echo "Copying brain file to home directory for access..."
    sudo cp /mnt/melvin_ssd/melvin/brain.m brain.m 2>/dev/null || cp /mnt/melvin_ssd/melvin/brain.m brain.m
    sudo chown melvin:melvin brain.m 2>/dev/null || chown melvin:melvin brain.m 2>/dev/null || true
fi

if [ ! -f brain.m ]; then
    echo "ERROR: Cannot find or access brain.m"
    exit 1
fi

if [ ! -x melvin_run_continuous ]; then
    echo "ERROR: melvin_run_continuous not found or not executable"
    exit 1
fi

echo "Starting Melvin..."
echo "Brain: $(pwd)/brain.m"
echo ""

./melvin_run_continuous brain.m 1
EOF

