#!/bin/bash
# Verify display is working with Spectre 24" 75Hz monitor

set -e

echo "=========================================="
echo "Spectre 24\" 75Hz Display Verification"
echo "=========================================="
echo ""

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "Step 1: Checking current display configuration..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    echo "Boot config:"
    sudo grep -E 'video=' /boot/extlinux/extlinux.conf | head -1
    echo ""
    echo "Current kernel cmdline:"
    cat /proc/cmdline | grep -o 'video=[^ ]*' || echo "No video parameter"
    echo ""
    echo "Framebuffer:"
    cat /sys/class/graphics/fb0/virtual_size 2>/dev/null || echo "No framebuffer"
EOF
echo ""

echo "Step 2: Testing console output..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    echo "╔════════════════════════════════════════════════════╗" | sudo tee /dev/tty1
    echo "║   MELVIN - Spectre 24\" 1080p @ 75Hz                ║" | sudo tee -a /dev/tty1
    echo "║   Display Test - $(date)                            ║" | sudo tee -a /dev/tty1
    echo "╚════════════════════════════════════════════════════╝" | sudo tee -a /dev/tty1
    echo "" | sudo tee -a /dev/tty1
    echo "If you see this on your Spectre display," | sudo tee -a /dev/tty1
    echo "the console output is working!" | sudo tee -a /dev/tty1
    echo "" | sudo tee -a /dev/tty1
    echo "Current settings:" | sudo tee -a /dev/tty1
    cat /proc/cmdline | grep -o 'video=[^ ]*' | sudo tee -a /dev/tty1 || echo "video=HDMI-A-1:1920x1080@75" | sudo tee -a /dev/tty1
    echo "" | sudo tee -a /dev/tty1
    sudo chvt 1
    echo "✓ Switched to console 1"
EOF
echo ""

echo "Step 3: Check if reboot is needed for 75Hz..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    CURRENT=$(cat /proc/cmdline | grep -o 'video=[^ ]*' | grep -o '@[0-9]*' || echo "@60")
    BOOT=$(sudo grep -E 'video=' /boot/extlinux/extlinux.conf | grep -o '@[0-9]*' | head -1 || echo "@60")
    
    if [ "$CURRENT" != "$BOOT" ]; then
        echo "⚠️  Reboot required!"
        echo "   Current: $CURRENT"
        echo "   Boot config: $BOOT"
        echo ""
        echo "To reboot now:"
        echo "  sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'sudo reboot'"
    else
        echo "✓ Refresh rate matches boot config"
    fi
EOF
echo ""

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "Check your Spectre 24\" display - you should see:"
echo "  - The test message above"
echo "  - Console output working"
echo ""
echo "If you see nothing:"
echo "  1. Check HDMI cable is connected"
echo "  2. Check display is powered on"
echo "  3. Try the other HDMI port on Jetson"
echo "  4. Reboot to apply 75Hz setting"
echo ""

