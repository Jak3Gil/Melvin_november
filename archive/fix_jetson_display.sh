#!/bin/bash
# Fix Jetson display port - enable HDMI/DP output

set -e

echo "=========================================="
echo "Fixing Jetson Display Port"
echo "=========================================="
echo ""

# Configuration
JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "Step 1: Checking current boot configuration..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo grep -E 'video|console|fbcon' /boot/extlinux/extlinux.conf | head -3"
echo ""

echo "Step 2: Backing up boot configuration..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo cp /boot/extlinux/extlinux.conf /boot/extlinux/extlinux.conf.backup.$(date +%Y%m%d_%H%M%S)"
echo "✓ Backup created"
echo ""

echo "Step 3: Enabling display port in boot configuration..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    # Find the active boot entry and add video parameter
    sudo sed -i '/^[[:space:]]*APPEND/s/$/ video=HDMI-A-1:1920x1080@60/' /boot/extlinux/extlinux.conf
    
    # Ensure console and fbcon are set
    sudo sed -i 's/console=tty0/console=tty0 fbcon=map:0/' /boot/extlinux/extlinux.conf
    
    # Enable DRM if not already enabled (remove modeset=0)
    sudo sed -i 's/nvidia-drm.modeset=0/nvidia-drm.modeset=1/' /boot/extlinux/extlinux.conf
    
    echo "✓ Boot configuration updated"
EOF
echo ""

echo "Step 4: Verifying changes..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo grep -E 'video|nvidia-drm.modeset' /boot/extlinux/extlinux.conf | head -3"
echo ""

echo "Step 5: Testing display output (before reboot)..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    # Test writing to console
    echo "Testing display output..." | sudo tee /dev/tty1
    echo "✓ Test message sent to display"
EOF
echo ""

echo "=========================================="
echo "Display Configuration Updated!"
echo "=========================================="
echo ""
echo "⚠️  REBOOT REQUIRED for changes to take effect"
echo ""
echo "To reboot now, run:"
echo "  sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'sudo reboot'"
echo ""
echo "After reboot, the display should show:"
echo "  - Console output on HDMI/DP port"
echo "  - Framebuffer at 1920x1080"
echo "  - Melvin can write to /dev/tty1 or /dev/fb0"
echo ""
echo "To test after reboot:"
echo "  sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'sudo chvt 1'"
echo "  sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'echo \"TEST\" | sudo tee /dev/tty1'"
echo ""

