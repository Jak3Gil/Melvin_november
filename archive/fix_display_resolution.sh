#!/bin/bash
# Fix Jetson Display Resolution - Run this script on the Jetson

set -e

echo "=== Jetson Display Resolution Fix ==="
echo ""

# Check current boot config
echo "Current boot configuration:"
sudo grep -E '^[[:space:]]*APPEND.*video' /boot/extlinux/extlinux.conf | head -1
echo ""

# Check current kernel cmdline
echo "Current kernel command line:"
cat /proc/cmdline | grep -o 'video=[^ ]*' || echo "No video parameter found"
echo ""

# Check framebuffer
echo "Current framebuffer resolution:"
if [ -f /sys/class/graphics/fb0/virtual_size ]; then
    cat /sys/class/graphics/fb0/virtual_size
else
    echo "Framebuffer not available"
fi
echo ""

# Show available resolutions to try
echo "Available resolution options:"
echo "  1. 640x480@60  (VGA - most compatible)"
echo "  2. 800x600@60  (SVGA - very compatible)"
echo "  3. 1024x768@60 (XGA - common)"
echo "  4. 1280x720@60 (HD - 720p)"
echo "  5. Auto-detect (let display choose)"
echo ""

read -p "Enter resolution choice (1-5) or press Enter to use 640x480@60: " choice
choice=${choice:-1}

case $choice in
    1)
        RESOLUTION="640x480@60"
        ;;
    2)
        RESOLUTION="800x600@60"
        ;;
    3)
        RESOLUTION="1024x768@60"
        ;;
    4)
        RESOLUTION="1280x720@60"
        ;;
    5)
        RESOLUTION=""
        ;;
    *)
        RESOLUTION="640x480@60"
        ;;
esac

echo ""
echo "Setting resolution to: ${RESOLUTION:-auto-detect}"
echo ""

# Backup original config
if [ ! -f /boot/extlinux/extlinux.conf.backup ]; then
    echo "Creating backup..."
    sudo cp /boot/extlinux/extlinux.conf /boot/extlinux/extlinux.conf.backup
fi

# Update boot config
if [ -z "$RESOLUTION" ]; then
    # Auto-detect mode
    sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1/' /boot/extlinux/extlinux.conf
else
    # Specific resolution
    sudo sed -i "s/video=HDMI-A-1[^ ]*/video=HDMI-A-1:${RESOLUTION}/" /boot/extlinux/extlinux.conf
fi

# Verify change
echo "Updated boot configuration:"
sudo grep -E '^[[:space:]]*APPEND.*video' /boot/extlinux/extlinux.conf | head -1
echo ""

read -p "Reboot now to apply changes? (y/n): " reboot_choice
if [ "$reboot_choice" = "y" ] || [ "$reboot_choice" = "Y" ]; then
    echo "Rebooting in 3 seconds..."
    sleep 3
    sudo reboot
else
    echo "Please reboot manually with: sudo reboot"
fi

