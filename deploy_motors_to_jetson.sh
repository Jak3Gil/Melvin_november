#!/bin/bash
#
# deploy_motors_to_jetson.sh - Deploy Motor Integration to Jetson
#
# Packages and deploys motor control system to Jetson via SSH
#

set -e

JETSON_USER="${JETSON_USER:-melvin}"
JETSON_HOST="${JETSON_HOST:-192.168.55.1}"
JETSON_PATH="/home/$JETSON_USER/melvin_motors"

echo "🚀 Deploying Motor Integration to Jetson"
echo "========================================="
echo ""
echo "Target: $JETSON_USER@$JETSON_HOST:$JETSON_PATH"
echo ""

# Check if we can reach Jetson
echo "📡 Checking Jetson connectivity..."
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo "❌ Cannot reach Jetson at $JETSON_HOST"
    echo ""
    echo "Make sure:"
    echo "  1. Jetson is powered on"
    echo "  2. Connected via USB or network"
    echo "  3. JETSON_HOST is correct (currently: $JETSON_HOST)"
    echo ""
    echo "Try:"
    echo "  export JETSON_HOST=<jetson-ip>"
    echo "  export JETSON_USER=<username>"
    exit 1
fi

echo "✅ Jetson reachable"

# Test SSH connection
echo ""
echo "🔐 Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 $JETSON_USER@$JETSON_HOST "echo 'SSH OK'" > /dev/null 2>&1; then
    echo "❌ Cannot SSH to Jetson"
    echo ""
    echo "Setup SSH key:"
    echo "  ssh-copy-id $JETSON_USER@$JETSON_HOST"
    exit 1
fi

echo "✅ SSH connection working"

# Create package directory
echo ""
echo "📦 Creating deployment package..."

PACKAGE_DIR="/tmp/melvin_motors_package"
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

# Copy source files
cp -r src $PACKAGE_DIR/
cp tools/map_can_motors.c $PACKAGE_DIR/
cp test_motor_exec.c $PACKAGE_DIR/
cp melvin_motor_runtime.c $PACKAGE_DIR/
cp setup_jetson_motors.sh $PACKAGE_DIR/
cp Makefile $PACKAGE_DIR/
cp tools/Makefile $PACKAGE_DIR/tools_Makefile

# Copy documentation
cp MOTOR_INTEGRATION.md $PACKAGE_DIR/
cp HARDWARE_INTEGRATION_REFINED.md $PACKAGE_DIR/ 2>/dev/null || true
cp README.md $PACKAGE_DIR/ 2>/dev/null || true

# Create deployment script for Jetson
cat > $PACKAGE_DIR/deploy.sh <<'EOF'
#!/bin/bash
# Jetson-side deployment script

set -e

echo "🔨 Compiling motor control tools..."

# Compile melvin.o first
gcc -O2 -std=c11 -Wall -Wextra -c -o melvin.o src/melvin.c -lm -lpthread

# Compile motor tools
echo "  - map_can_motors"
gcc -O2 -std=c11 -Wall -Wextra -o tools/map_can_motors map_can_motors.c melvin.o -lm -lpthread

echo "  - test_motor_exec"  
gcc -O2 -std=c11 -Wall -Wextra -o test_motor_exec test_motor_exec.c melvin.o -lm -lpthread

echo "  - melvin_motor_runtime"
gcc -O2 -std=c11 -Wall -Wextra -o melvin_motor_runtime melvin_motor_runtime.c melvin.o -lm -lpthread

echo "✅ Compilation complete"
echo ""
echo "Next steps:"
echo "  1. Create/copy brain file to this directory"
echo "  2. Run: sudo ./setup_jetson_motors.sh brain.m"
echo "  3. Test: sudo ./test_motor_exec brain.m all"
echo "  4. Start: sudo ./melvin_motor_runtime brain.m"
echo ""
echo "See MOTOR_INTEGRATION.md for full documentation"
EOF

chmod +x $PACKAGE_DIR/deploy.sh
chmod +x $PACKAGE_DIR/setup_jetson_motors.sh

# Create tarball
echo "  Creating tarball..."
cd /tmp
tar czf melvin_motors_package.tar.gz melvin_motors_package/
echo "✅ Package created: $(du -h melvin_motors_package.tar.gz | cut -f1)"

# Copy to Jetson
echo ""
echo "📤 Uploading to Jetson..."

ssh $JETSON_USER@$JETSON_HOST "mkdir -p $JETSON_PATH"
scp /tmp/melvin_motors_package.tar.gz $JETSON_USER@$JETSON_HOST:/tmp/

echo "✅ Upload complete"

# Extract on Jetson
echo ""
echo "📦 Extracting on Jetson..."

ssh $JETSON_USER@$JETSON_HOST <<REMOTE_EOF
cd /tmp
tar xzf melvin_motors_package.tar.gz
rm -rf $JETSON_PATH/*
cp -r melvin_motors_package/* $JETSON_PATH/
cd $JETSON_PATH
./deploy.sh
REMOTE_EOF

echo "✅ Extraction and compilation complete"

# Copy brain file if it exists
BRAIN_FILE="brain_teachable.m"
if [ -f "$BRAIN_FILE" ]; then
    echo ""
    echo "📤 Copying brain file..."
    scp $BRAIN_FILE $JETSON_USER@$JETSON_HOST:$JETSON_PATH/
    echo "✅ Brain file copied"
else
    echo ""
    echo "⚠️  No brain file found locally ($BRAIN_FILE)"
    echo "   You'll need to create one on the Jetson or copy it manually"
fi

# Summary
echo ""
echo "🎉 Deployment Complete!"
echo "======================="
echo ""
echo "Files deployed to: $JETSON_USER@$JETSON_HOST:$JETSON_PATH"
echo ""
echo "To setup motors on Jetson, SSH in and run:"
echo "  ssh $JETSON_USER@$JETSON_HOST"
echo "  cd $JETSON_PATH"
echo "  sudo ./setup_jetson_motors.sh brain.m"
echo ""
echo "Or run remotely:"
echo "  ssh $JETSON_USER@$JETSON_HOST 'cd $JETSON_PATH && sudo ./setup_jetson_motors.sh brain.m'"
echo ""
echo "View logs:"
echo "  ssh $JETSON_USER@$JETSON_HOST 'tail -f $JETSON_PATH/*.log'"
echo ""
echo "See MOTOR_INTEGRATION.md for full documentation"
echo ""

# Cleanup
rm -rf $PACKAGE_DIR
rm -f /tmp/melvin_motors_package.tar.gz

echo "✅ Local cleanup complete"

