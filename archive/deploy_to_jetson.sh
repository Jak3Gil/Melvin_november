#!/bin/bash
# Deploy melvin to Jetson via USB/SSH

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${1:-jetson.local}"  # Default to jetson.local, or provide IP/hostname
JETSON_DIR="~/melvin"

echo "=========================================="
echo "Deploying Melvin to Jetson"
echo "=========================================="
echo "Target: ${JETSON_USER}@${JETSON_HOST}"
echo "Directory: ${JETSON_DIR}"
echo ""

# Check if we can connect
echo "Testing connection..."
if ! sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${JETSON_USER}@${JETSON_HOST}" "echo 'Connection successful'" 2>/dev/null; then
    echo "ERROR: Cannot connect to Jetson"
    echo "Trying to find USB mount..."
    
    # Try to find USB mount
    USB_MOUNT=$(df -h | grep -i "jetson\|usb" | awk '{print $NF}' | head -1)
    if [ -n "$USB_MOUNT" ] && [ -d "$USB_MOUNT" ]; then
        echo "Found USB mount at: $USB_MOUNT"
        JETSON_DIR="$USB_MOUNT/melvin"
        mkdir -p "$JETSON_DIR"
        echo "Copying files via USB..."
        cp melvin.c melvin.h "$JETSON_DIR/"
        if [ -f melvin.m ]; then
            cp melvin.m "$JETSON_DIR/"
        fi
        echo "✓ Files copied to $JETSON_DIR"
        echo ""
        echo "Next steps:"
        echo "1. Eject USB and plug into Jetson"
        echo "2. On Jetson, run: cd $JETSON_DIR && ./setup_jetson.sh"
        exit 0
    else
        echo "ERROR: Cannot connect via SSH or find USB mount"
        echo "Please ensure:"
        echo "  - Jetson is on same network (use: sshpass -p '$JETSON_PASS' ssh ${JETSON_USER}@<IP>)"
        echo "  - Or USB is mounted"
        exit 1
    fi
fi

echo "✓ Connection successful"
echo ""

# Install sshpass if needed
if ! command -v sshpass &> /dev/null; then
    echo "Installing sshpass..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install hudochenkov/sshpass/sshpass
        else
            echo "ERROR: sshpass not found. Install with: brew install hudochenkov/sshpass/sshpass"
            exit 1
        fi
    else
        sudo apt-get install -y sshpass || sudo yum install -y sshpass
    fi
fi

# Create directory on Jetson
echo "Creating directory on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" \
    "mkdir -p ${JETSON_DIR} && mkdir -p ${JETSON_DIR}/plugins"

# Copy core files
echo "Copying core files..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    melvin.c melvin.h "${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/"

# Copy brain file if it exists
if [ -f melvin.m ]; then
    echo "Copying brain file (melvin.m)..."
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        melvin.m "${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/"
else
    echo "No melvin.m found - will be created on Jetson"
fi

# Copy startup/monitor scripts
echo "Copying scripts..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    start_melvin.sh monitor_melvin.sh stop_melvin.sh \
    "${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/"

# Copy monitor binary if exists
if [ -f monitor_melvin ]; then
    echo "Copying monitor binary..."
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        monitor_melvin "${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/"
fi

# Copy init script if exists
if [ -f init_melvin_simple ]; then
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        init_melvin_simple "${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/"
fi

# Create setup script on Jetson
echo "Creating setup script on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" "cat > ${JETSON_DIR}/setup_jetson.sh << 'EOF'
#!/bin/bash
cd ~/melvin

# Compile melvin
echo 'Compiling melvin...'
gcc -o melvin melvin.c -lm -ldl

# Compile monitor if source exists
if [ -f monitor_melvin.c ]; then
    echo 'Compiling monitor...'
    gcc -o monitor_melvin monitor_melvin.c -lm
fi

# Initialize brain if needed
if [ ! -f melvin.m ]; then
    echo 'Initializing brain...'
    if [ -f init_melvin_simple ]; then
        ./init_melvin_simple
    else
        echo 'ERROR: No init script found'
    fi
fi

# Make scripts executable
chmod +x *.sh

echo ''
echo 'Setup complete!'
echo 'Start melvin: ./start_melvin.sh'
echo 'Monitor: ./monitor_melvin.sh'
EOF
chmod +x ${JETSON_DIR}/setup_jetson.sh"

echo ""
echo "=========================================="
echo "Deployment Complete"
echo "=========================================="
echo ""
echo "Next steps on Jetson:"
echo "  ssh ${JETSON_USER}@${JETSON_HOST}"
echo "  cd ${JETSON_DIR}"
echo "  ./setup_jetson.sh"
echo "  ./start_melvin.sh"
echo ""
echo "Then monitor from here:"
echo "  ./monitor_jetson.sh ${JETSON_HOST}"
echo ""

