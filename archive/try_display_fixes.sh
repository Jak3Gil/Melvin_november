#!/bin/bash
# Try different display resolutions based on online research for Jetson Orin AGX

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "=== Jetson Orin AGX Display Fix Script ==="
echo "Based on NVIDIA documentation and forum solutions"
echo ""

# Function to apply and reboot
apply_and_reboot() {
    local resolution=$1
    local description=$2
    
    echo "Trying: $description"
    echo "Setting: $resolution"
    
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
        "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=$resolution/' /boot/extlinux/extlinux.conf && \
         sudo grep video /boot/extlinux/extlinux.conf | head -1"
    
    read -p "Reboot now to test this resolution? (y/n/s=skip): " choice
    case $choice in
        y|Y)
            echo "Rebooting..."
            sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "sudo reboot"
            echo "Wait 30-60 seconds for reboot, then check display."
            echo "If it works, great! If not, run this script again and try the next option."
            exit 0
            ;;
        s|S)
            echo "Skipping reboot - you can reboot manually later"
            return
            ;;
        *)
            echo "Not rebooting - you can test this later"
            return
            ;;
    esac
}

echo "Select resolution to try:"
echo "1. Auto-detect (let EDID negotiate) - RECOMMENDED FIRST"
echo "2. 1280x720@60 (HD 720p) - Most compatible HD"
echo "3. 1024x768@60 (XGA) - Very common"
echo "4. 800x600@60 (SVGA) - Very compatible"
echo "5. 640x480@60 (VGA) - Universal support"
echo "6. 1280x720@50 (HD with lower refresh)"
echo "7. Try DisplayPort instead (DP-0:1280x720@60)"
echo "8. Custom resolution"
echo ""

read -p "Enter choice (1-8): " choice

case $choice in
    1)
        apply_and_reboot "HDMI-A-1" "Auto-detect (EDID negotiation)"
        ;;
    2)
        apply_and_reboot "HDMI-A-1:1280x720@60" "HD 720p @ 60Hz"
        ;;
    3)
        apply_and_reboot "HDMI-A-1:1024x768@60" "XGA @ 60Hz"
        ;;
    4)
        apply_and_reboot "HDMI-A-1:800x600@60" "SVGA @ 60Hz"
        ;;
    5)
        apply_and_reboot "HDMI-A-1:640x480@60" "VGA @ 60Hz"
        ;;
    6)
        apply_and_reboot "HDMI-A-1:1280x720@50" "HD 720p @ 50Hz"
        ;;
    7)
        apply_and_reboot "DP-0:1280x720@60" "DisplayPort HD 720p @ 60Hz"
        ;;
    8)
        read -p "Enter resolution (format: HDMI-A-1:WIDTHxHEIGHT@RATE): " custom_res
        apply_and_reboot "$custom_res" "Custom: $custom_res"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Current boot configuration:"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "sudo grep video /boot/extlinux/extlinux.conf | head -1"

echo ""
echo "To apply changes, reboot with:"
echo "  sshpass -p '$JETSON_PASS' ssh $JETSON_USER@$JETSON_IP 'sudo reboot'"

