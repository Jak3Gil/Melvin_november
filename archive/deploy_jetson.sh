#!/bin/bash
# Deploy Melvin to Jetson via USB connection
# Only transfers melvin.m, melvin.c, melvin.h - everything else is in the graph

set -e

echo "=========================================="
echo "Deploying Melvin to Jetson"
echo "=========================================="

# Connection parameters
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST=""  # Will be detected
JETSON_DIR="~/melvin_system"

# Try different connection methods
echo "Detecting Jetson connection..."

# Method 1: Try direct ethernet IP (if configured)
if ping -c 1 -W 1 169.254.123.100 &>/dev/null; then
    JETSON_HOST="169.254.123.100"
    echo "✓ Found Jetson at $JETSON_HOST (ethernet)"
elif ping -c 1 -W 1 192.168.55.1 &>/dev/null; then
    JETSON_HOST="192.168.55.1"
    echo "✓ Found Jetson at $JETSON_HOST (USB network)"
else
    # Method 2: Try USB serial devices
    for dev in /dev/cu.usb* /dev/tty.usb*; do
        if [ -e "$dev" ]; then
            echo "Found USB device: $dev"
            echo "Attempting serial connection..."
            # Serial connection would need different approach (screen/minicom)
            # For now, prompt for IP
            read -p "Enter Jetson IP address (or press Enter to use $dev): " JETSON_HOST
            if [ -z "$JETSON_HOST" ]; then
                JETSON_HOST="$dev"
            fi
            break
        fi
    done
fi

# If still no host, prompt
if [ -z "$JETSON_HOST" ] || [[ "$JETSON_HOST" == /dev/* ]]; then
    echo ""
    echo "Could not auto-detect Jetson. Please choose:"
    echo "1. Enter IP address (if Jetson is on network)"
    echo "2. Use USB serial device (manual connection required)"
    echo ""
    read -p "Enter Jetson IP address (or 'serial' for manual): " JETSON_HOST
    
    if [ -z "$JETSON_HOST" ]; then
        echo "No host specified. Exiting."
        exit 1
    fi
    
    if [ "$JETSON_HOST" = "serial" ]; then
        echo ""
        echo "For USB serial connection, use:"
        echo "  screen /dev/cu.usbserial-* 115200"
        echo "  (or check /dev/tty.usb* devices)"
        echo ""
        echo "Then manually copy files using:"
        echo "  scp melvin.c melvin.h melvin.m melvin@JETSON_IP:~/melvin_system/"
        exit 0
    fi
fi

# Files to transfer (only the essentials)
FILES=("melvin.c" "melvin.h" "melvin.m")

echo ""
echo "Files to deploy:"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file (NOT FOUND!)"
        exit 1
    fi
done

echo ""
echo "Target: $JETSON_USER@$JETSON_HOST:$JETSON_DIR"

# Check if sshpass is available (for password authentication)
if command -v sshpass &> /dev/null; then
    SSH_CMD="sshpass -p '$JETSON_PASS' ssh"
    SCP_CMD="sshpass -p '$JETSON_PASS' scp"
else
    echo ""
    echo "Warning: sshpass not found. You may need to enter password manually."
    echo "Install with: brew install hudochenkov/sshpass/sshpass"
    echo ""
    SSH_CMD="ssh"
    SCP_CMD="scp"
fi

# Create directory on Jetson
echo ""
echo "Creating directory on Jetson..."
$SSH_CMD -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_HOST" "mkdir -p $JETSON_DIR" || {
    echo "Failed to connect. Trying without password..."
    ssh "$JETSON_USER@$JETSON_HOST" "mkdir -p $JETSON_DIR"
}

# Transfer files
echo ""
echo "Transferring files..."
for file in "${FILES[@]}"; do
    echo "  → $file"
    $SCP_CMD -o StrictHostKeyChecking=no "$file" "$JETSON_USER@$JETSON_HOST:$JETSON_DIR/" || {
        echo "Failed with sshpass, trying regular scp..."
        scp "$file" "$JETSON_USER@$JETSON_HOST:$JETSON_DIR/"
    }
done

echo ""
echo "✓ Files transferred!"
echo ""
echo "Next steps on Jetson:"
echo "  1. ssh $JETSON_USER@$JETSON_HOST"
echo "  2. cd ~/melvin_system"
echo "  3. Build: gcc -std=c11 -O3 -o melvin melvin.c -lm -ldl -lpthread"
echo "  4. Run: ./melvin"
echo ""

# Optionally, build and run remotely
read -p "Build and run on Jetson now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Building on Jetson..."
    $SSH_CMD "$JETSON_USER@$JETSON_HOST" "cd $JETSON_DIR && gcc -std=c11 -O3 -o melvin melvin.c -lm -ldl -lpthread" || {
        echo "Build failed. Check Jetson manually."
        exit 1
    }
    
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "To run Melvin on Jetson:"
    echo "  ssh $JETSON_USER@$JETSON_HOST"
    echo "  cd ~/melvin_system"
    echo "  ./melvin"
    echo ""
fi

echo "=========================================="
echo "Deployment complete!"
echo "=========================================="

