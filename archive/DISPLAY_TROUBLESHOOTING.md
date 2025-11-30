# Display Troubleshooting Guide

## Current Status
- Boot config: `video=HDMI-A-1:640x480@60` (or `HDMI-A-2:640x480@60` if trying alternate port)
- Kernel cmdline shows the video parameter is being applied
- Framebuffer may show different resolution than actual output

## "Operation Not Permitted" Errors

If you see "operation not permitted" errors, try:

### 1. Check Permissions
```bash
# Make sure you're in the right groups
groups
# Should include: video, render, etc.

# Try with sudo
sudo <command>
```

### 2. Common Permission Issues

**Accessing framebuffer:**
```bash
# Check permissions
ls -la /dev/fb0
# If needed, add user to video group
sudo usermod -a -G video melvin
```

**Accessing DRM devices:**
```bash
# Check permissions
ls -la /dev/dri/
# Add to render group if needed
sudo usermod -a -G render melvin
```

**Reading kernel messages:**
```bash
# dmesg requires sudo
sudo dmesg | grep -i display
```

## If Display Still Shows "Signal Out of Range"

### Try Different Resolutions

1. **640x480@60** (VGA - most compatible) - CURRENTLY SET
2. **800x600@60** (SVGA)
3. **1024x768@60** (XGA)
4. **1280x720@60** (720p HD)
5. **Auto-detect** (remove resolution, let display choose)

### Try Different HDMI Ports

The Jetson may have multiple HDMI outputs:
- `HDMI-A-1` (default)
- `HDMI-A-2` (alternate)

To switch:
```bash
sudo sed -i 's/video=HDMI-A-1/video=HDMI-A-2/' /boot/extlinux/extlinux.conf
sudo reboot
```

### Try Different Refresh Rates

Some displays don't support 60Hz at certain resolutions:
```bash
# Try 50Hz instead
sudo sed -i 's/@60/@50/' /boot/extlinux/extlinux.conf
sudo reboot
```

### Hardware Checks

1. **Check HDMI cable** - Try a different cable
2. **Check display** - Try a different monitor/TV
3. **Check HDMI port** - Try the other HDMI port on Jetson
4. **Check display input** - Make sure monitor is set to correct HDMI input

## Quick Fix Commands

### Set to 640x480@60 (most compatible)
```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:640x480@60/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Set to auto-detect
```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Try alternate HDMI port
```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/HDMI-A-1/HDMI-A-2/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Check current settings
```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "cat /proc/cmdline | grep video && sudo grep video /boot/extlinux/extlinux.conf | head -1"
```

## Verify Display is Working

After reboot, check:
1. Does the display show anything? (even if wrong resolution)
2. Is there still "signal out of range"?
3. Can you see console output?

If you see console output but wrong resolution, the display is working - we just need to find the right resolution.

## Next Steps

1. Try the script: `fix_display_resolution.sh` (copy to Jetson and run)
2. Try different resolutions one by one
3. Try alternate HDMI port
4. Check hardware (cable, monitor)

