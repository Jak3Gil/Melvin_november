#!/bin/bash
# enable_jetson_can.sh - Enable CAN on Jetson AGX Orin

echo "╔═══════════════════════════════════════════╗"
echo "║  Enable CAN on Jetson AGX Orin            ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

echo "The CAN pins are UNCLAIMED in the device tree."
echo "We need to enable them using jetson-io."
echo ""
echo "═══════════════════════════════════════════"
echo ""

# Method 1: Try jetson-io tool
if [ -f /opt/nvidia/jetson-io/jetson-io.py ]; then
    echo "Option 1: Interactive Configuration"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Run this command to enable CAN:"
    echo "  sudo /opt/nvidia/jetson-io/jetson-io.py"
    echo ""
    echo "Then:"
    echo "  1. Select 'Configure Jetson 40pin Header'"
    echo "  2. Find and enable 'CAN0' or 'mttcan0'"
    echo "  3. Save and reboot"
    echo ""
fi

# Method 2: Check current configuration
echo ""
echo "Option 2: Check Current Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f /boot/dtb/kernel_tegra234-p3701-0000-p3737-0000.dtb ]; then
    echo "✅ Device tree found"
    
    # Check if CAN is enabled in current DTB
    if fdtdump /boot/dtb/kernel_tegra234-p3701-0000-p3737-0000.dtb 2>/dev/null | grep -q mttcan; then
        echo "✅ MTTCAN present in device tree"
    else
        echo "⚠️  MTTCAN may need enabling"
    fi
fi

echo ""
echo "Option 3: Manual Device Tree Overlay"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Create file: /boot/tegra234-p3737-camera-overlay-CAN.dts"
echo ""
cat << 'EOF'
/dts-v1/;
/plugin/;

/ {
    overlay-name = "Enable CAN0";
    compatible = "nvidia,tegra234";
    
    fragment@0 {
        target-path = "/";
        __overlay__ {
            mttcan@c310000 {
                status = "okay";
            };
        };
    };
};
EOF

echo ""
echo "Then compile and apply:"
echo "  sudo dtc -I dts -O dtb -o /boot/CAN.dtbo tegra234-CAN.dts"
echo "  sudo reboot"
echo ""

echo "═══════════════════════════════════════════"
echo ""
echo "🎯 RECOMMENDED: Use jetson-io tool (Option 1)"
echo ""
echo "After enabling CAN and rebooting, check:"
echo "  cat /sys/kernel/debug/pinctrl/*/pinmux-pins | grep CAN0"
echo ""
echo "Should see:"
echo "  pin 138 (CAN0_DOUT): mttcan0 (CLAIMED)"
echo "  pin 139 (CAN0_DIN):  mttcan0 (CLAIMED)"
echo ""
echo "Then CAN will work! 🚀"

