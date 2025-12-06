#!/bin/bash
#
# deploy_usb_can_to_jetson.sh - Deploy USB-to-CAN motor setup to Jetson via USB
#
# This script:
# 1. Connects to Jetson via USB (using jetson_terminal.sh or direct USB)
# 2. Copies setup script and motor test code
# 3. Runs setup on Jetson
# 4. Tests motors 12 and 14
#

set -e

echo "╔═══════════════════════════════════════════╗"
echo "║  Deploy USB-to-CAN Setup to Jetson        ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Configuration
JETSON_USER="${JETSON_USER:-melvin}"
JETSON_HOST="${JETSON_HOST:-192.168.55.1}"
JETSON_PATH="/home/$JETSON_USER/melvin_motors"

# Check connection method
echo "🔍 Detecting Jetson connection method..."
echo ""

# Method 1: Try SSH (USB networking)
if ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo "✅ Jetson reachable via network: $JETSON_HOST"
    CONNECTION_METHOD="ssh"
elif [ -f "jetson_terminal.sh" ]; then
    echo "✅ Found jetson_terminal.sh - will use USB serial"
    CONNECTION_METHOD="serial"
else
    echo "⚠️  Cannot detect Jetson connection"
    echo ""
    echo "Options:"
    echo "  1. Set JETSON_HOST for SSH: export JETSON_HOST=192.168.55.1"
    echo "  2. Use USB serial: ./jetson_terminal.sh"
    echo ""
    read -p "Continue with manual setup? (y/n): " -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
    CONNECTION_METHOD="manual"
fi

# Create deployment package
echo ""
echo "📦 Creating deployment package..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PACKAGE_DIR="/tmp/melvin_usb_can_package"
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

# Copy essential files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "$SCRIPT_DIR/setup_usb_can_motors.sh" "$PACKAGE_DIR/"
chmod +x "$PACKAGE_DIR/setup_usb_can_motors.sh"

# Copy motor test code
if [ -f "$SCRIPT_DIR/test_motors_12_14.c" ]; then
    cp "$SCRIPT_DIR/test_motors_12_14.c" "$PACKAGE_DIR/"
elif [ -f "$SCRIPT_DIR/test_motor_12_14.c" ]; then
    cp "$SCRIPT_DIR/test_motor_12_14.c" "$PACKAGE_DIR/"
    # Rename for consistency
    cp "$SCRIPT_DIR/test_motor_12_14.c" "$PACKAGE_DIR/test_motors_12_14.c"
else
    echo "   ⚠️  Motor test source not found"
    echo "   Will need to compile on Jetson"
fi

# Create quick start script
cat > "$PACKAGE_DIR/quick_start.sh" <<'EOF'
#!/bin/bash
# Quick start script for Jetson

echo "🚀 Quick Start - USB-to-CAN Motors"
echo "===================================="
echo ""

# Run setup
sudo ./setup_usb_can_motors.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "To test motors again:"
echo "  sudo ./test_motors_12_14"
echo ""
EOF

chmod +x "$PACKAGE_DIR/quick_start.sh"

# Create tarball
echo "   Creating tarball..."
cd /tmp
tar czf melvin_usb_can_package.tar.gz melvin_usb_can_package/ 2>/dev/null
echo "   ✅ Package created"

# Deploy based on connection method
if [ "$CONNECTION_METHOD" = "ssh" ]; then
    echo ""
    echo "📤 Deploying via SSH..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Test SSH
    if ! ssh -o ConnectTimeout=5 $JETSON_USER@$JETSON_HOST "echo 'SSH OK'" > /dev/null 2>&1; then
        echo "❌ Cannot SSH to Jetson"
        echo ""
        echo "Setup SSH key:"
        echo "  ssh-copy-id $JETSON_USER@$JETSON_HOST"
        exit 1
    fi
    
    # Create directory on Jetson
    ssh $JETSON_USER@$JETSON_HOST "mkdir -p $JETSON_PATH"
    
    # Copy package
    scp /tmp/melvin_usb_can_package.tar.gz $JETSON_USER@$JETSON_HOST:/tmp/
    
    # Extract and setup
    ssh $JETSON_USER@$JETSON_HOST <<REMOTE_EOF
cd /tmp
tar xzf melvin_usb_can_package.tar.gz
cp -r melvin_usb_can_package/* $JETSON_PATH/
cd $JETSON_PATH
chmod +x *.sh
echo ""
echo "✅ Files deployed to $JETSON_PATH"
echo ""
echo "To run setup:"
echo "  cd $JETSON_PATH"
echo "  sudo ./setup_usb_can_motors.sh"
REMOTE_EOF
    
    echo ""
    echo "✅ Deployment complete via SSH"
    echo ""
    echo "To run setup on Jetson:"
    echo "  ssh $JETSON_USER@$JETSON_HOST"
    echo "  cd $JETSON_PATH"
    echo "  sudo ./setup_usb_can_motors.sh"
    
elif [ "$CONNECTION_METHOD" = "serial" ]; then
    echo ""
    echo "📤 Deployment via USB serial..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "⚠️  USB serial deployment requires manual steps"
    echo ""
    echo "1. Connect to Jetson via USB serial:"
    echo "   ./jetson_terminal.sh"
    echo ""
    echo "2. On Jetson, create directory:"
    echo "   mkdir -p $JETSON_PATH"
    echo ""
    echo "3. Copy files manually or use scp from another method"
    echo ""
    echo "Package is ready at: /tmp/melvin_usb_can_package.tar.gz"
    echo ""
    
else
    echo ""
    echo "📦 Manual Deployment"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Package created at: /tmp/melvin_usb_can_package/"
    echo ""
    echo "To deploy manually:"
    echo "  1. Copy package to Jetson (USB drive, scp, etc.)"
    echo "  2. Extract: tar xzf melvin_usb_can_package.tar.gz"
    echo "  3. Run: sudo ./setup_usb_can_motors.sh"
    echo ""
fi

# Cleanup
echo ""
read -p "Keep package for manual deployment? (y/n): " -r keep
if [[ ! "$keep" =~ ^[Yy]$ ]]; then
    rm -rf $PACKAGE_DIR
    rm -f /tmp/melvin_usb_can_package.tar.gz
    echo "✅ Cleanup complete"
else
    echo "✅ Package kept at: /tmp/melvin_usb_can_package/"
fi

echo ""
echo "🎉 Deployment ready!"
echo ""

