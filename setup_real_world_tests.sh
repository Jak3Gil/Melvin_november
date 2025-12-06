#!/bin/bash
# Setup script for real-world testing on Jetson
# - USB camera detection
# - GPU/CPU EXEC node setup
# - Real data pipeline

set -e

echo "=== Real-World Test Setup for Jetson ==="
echo ""

# Check for USB cameras
echo "Detecting USB cameras..."
CAMERAS=$(ls /dev/video* 2>/dev/null | head -5)
if [ -z "$CAMERAS" ]; then
    echo "⚠ No USB cameras found"
    echo "  Install: sudo apt-get install v4l-utils"
    echo "  List: v4l2-ctl --list-devices"
else
    echo "✓ Found cameras:"
    for cam in $CAMERAS; do
        echo "  $cam"
        if command -v v4l2-ctl >/dev/null 2>&1; then
            v4l2-ctl -d "$cam" --all 2>/dev/null | grep -E "Card type|Driver name" | head -2
        fi
    done
fi
echo ""

# Check GPU availability
echo "Checking GPU (CUDA/OpenCL)..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
elif command -v clinfo >/dev/null 2>&1; then
    echo "✓ OpenCL available:"
    clinfo -l 2>/dev/null | head -5
else
    echo "⚠ No GPU detected (will use CPU fallback)"
fi
echo ""

# Check required libraries
echo "Checking required libraries..."
MISSING=""

if ! pkg-config --exists v4l2 2>/dev/null; then
    MISSING="$MISSING libv4l2-dev"
fi

if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
    MISSING="$MISSING libopencv-dev"
fi

if [ -n "$MISSING" ]; then
    echo "⚠ Missing libraries:$MISSING"
    echo "  Install: sudo apt-get install$MISSING"
else
    echo "✓ All libraries available"
fi
echo ""

# Create test directory structure
echo "Creating test directories..."
mkdir -p real_world_tests/{frames,output,logs}
echo "✓ Directories created"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Connect USB camera to Jetson"
echo "  2. Run: ./test_real_world /dev/video0"
echo "  3. Check real_world_tests/ for output"

