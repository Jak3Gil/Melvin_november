#!/bin/bash
#
# deploy_motor_test.sh - Quick deploy motor test to Jetson
#

JETSON_USER="${JETSON_USER:-melvin}"
JETSON_HOST="${JETSON_HOST:-192.168.55.1}"

echo "🚀 Deploying motor test to Jetson"
echo "Target: $JETSON_USER@$JETSON_HOST"
echo ""

# Check connectivity
if ! ping -c 1 -W 2 $JETSON_HOST > /dev/null 2>&1; then
    echo "❌ Cannot reach Jetson at $JETSON_HOST"
    exit 1
fi

echo "📤 Uploading test files..."

# Copy test files
scp test_motor_12_14.c $JETSON_USER@$JETSON_HOST:/tmp/
scp test_motors_interactive.sh $JETSON_USER@$JETSON_HOST:/tmp/

echo "✅ Upload complete"
echo ""
echo "To run on Jetson:"
echo "  ssh $JETSON_USER@$JETSON_HOST"
echo "  cd /tmp"
echo "  sudo ./test_motors_interactive.sh"
echo ""
echo "Or run remotely:"
echo "  ssh -t $JETSON_USER@$JETSON_HOST 'cd /tmp && sudo ./test_motors_interactive.sh'"

