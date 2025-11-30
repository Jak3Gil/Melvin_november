#!/bin/bash
# Manual deployment - specify Jetson IP

if [ -z "$1" ]; then
    echo "Usage: $0 <jetson_ip>"
    echo ""
    echo "Example: $0 192.168.1.50"
    echo ""
    echo "To find Jetson IP:"
    echo "  - Check Jetson display"
    echo "  - On Jetson: hostname -I"
    echo "  - Check router admin page"
    exit 1
fi

JETSON_IP="$1"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_DIR="~/melvin"

echo "=========================================="
echo "Deploying Melvin to Jetson"
echo "=========================================="
echo "IP: $JETSON_IP"
echo "User: $JETSON_USER"
echo ""

# Test connection
echo "Testing connection..."
if ! sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${JETSON_USER}@${JETSON_IP}" "echo 'Connected'" 2>/dev/null; then
    echo "ERROR: Cannot connect to $JETSON_IP"
    echo ""
    echo "Please check:"
    echo "  1. Jetson is powered on"
    echo "  2. Jetson is on same network"
    echo "  3. SSH is enabled: sudo systemctl enable ssh"
    echo "  4. IP is correct: hostname -I (on Jetson)"
    echo ""
    echo "Try connecting manually:"
    echo "  ssh ${JETSON_USER}@${JETSON_IP}"
    exit 1
fi

echo "✓ Connection successful"
echo ""

# Create directory
echo "Creating directory on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_IP}" \
    "mkdir -p ${JETSON_DIR} && mkdir -p ${JETSON_DIR}/plugins && mkdir -p ${JETSON_DIR}/ingested_repos"

# Copy files
echo "Copying files..."
echo "  → melvin.c"
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    melvin.c "${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/"

echo "  → melvin.h"
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    melvin.h "${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/"

if [ -f melvin.m ]; then
    echo "  → melvin.m (brain file - 29MB, may take a moment...)"
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        melvin.m "${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/"
fi

echo "  → Scripts"
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    start_melvin.sh monitor_melvin.sh stop_melvin.sh \
    "${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/" 2>/dev/null || echo "    (some scripts may not exist)"

if [ -f monitor_melvin.c ]; then
    echo "  → monitor_melvin.c"
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        monitor_melvin.c "${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/"
fi

if [ -f init_melvin_simple.c ]; then
    echo "  → init_melvin_simple.c"
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        init_melvin_simple.c "${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/"
fi

# Create setup script
echo "Creating setup script..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_IP}" "cat > ${JETSON_DIR}/setup_jetson.sh << 'EOF'
#!/bin/bash
cd ~/melvin

echo '=========================================='
echo 'Setting up Melvin on Jetson'
echo '=========================================='

# Compile melvin
echo 'Compiling melvin...'
gcc -o melvin melvin.c -lm -ldl 2>&1 | grep -E '(error|warning)' || echo '✓ Compiled'

# Compile monitor
if [ -f monitor_melvin.c ]; then
    echo 'Compiling monitor...'
    gcc -o monitor_melvin monitor_melvin.c -lm 2>&1 | grep -E '(error|warning)' || echo '✓ Monitor compiled'
fi

# Initialize brain if needed
if [ ! -f melvin.m ]; then
    echo 'Initializing brain...'
    if [ -f init_melvin_simple.c ]; then
        gcc -o init_melvin_simple init_melvin_simple.c -lm
        ./init_melvin_simple
    else
        echo 'Creating minimal brain...'
        # Create 29MB brain file manually
        dd if=/dev/zero of=melvin.m bs=1M count=29 2>/dev/null
        echo '⚠ Brain file created but not initialized - melvin will initialize on first run'
    fi
fi

# Make scripts executable
chmod +x *.sh 2>/dev/null

echo ''
echo '=========================================='
echo 'Setup Complete!'
echo '=========================================='
echo ''
echo 'To start Melvin:'
echo '  ./start_melvin.sh'
echo ''
echo 'To monitor:'
echo '  ./monitor_melvin.sh'
echo ''
EOF
chmod +x ${JETSON_DIR}/setup_jetson.sh"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps on Jetson:"
echo "  ssh ${JETSON_USER}@${JETSON_IP}"
echo "  cd ~/melvin"
echo "  ./setup_jetson.sh"
echo "  ./start_melvin.sh"
echo ""
echo "Then monitor from Mac:"
echo "  ./monitor_jetson.sh ${JETSON_IP}"
echo ""


