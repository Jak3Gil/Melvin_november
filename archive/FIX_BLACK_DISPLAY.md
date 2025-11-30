# Fix Black Display on Jetson

## Problem

The display port shows black because the framebuffer device (`/dev/fb0`) is not available.

## Solutions

### Option 1: Enable Framebuffer Console

On the Jetson, enable framebuffer console:

```bash
# SSH to Jetson
ssh melvin@169.254.123.100

# Check boot configuration
cat /boot/extlinux/extlinux.conf | grep -i console

# Enable framebuffer console (edit extlinux.conf)
sudo nano /boot/extlinux/extlinux.conf
# Add or modify: console=tty0 console=ttyS0,115200n8 fbcon=map:0

# Reboot
sudo reboot
```

### Option 2: Check Display Connection

1. **Verify HDMI/eDP cable is connected**
2. **Check display is powered on**
3. **Try different display port** (HDMI vs eDP)

### Option 3: Use X11/Wayland

If X11 is available, we can output to X display:

```bash
# Check if X server is running
ps aux | grep X

# If not, start X server (headless mode)
X :0 &
export DISPLAY=:0
```

### Option 4: Use Simple Text Console

For now, the display shows text output which works without framebuffer:

```bash
# View on Jetson console (if connected via serial/USB)
# Or redirect to text file and view remotely
ssh melvin@169.254.123.100 "tail -f ~/melvin_system/melvin.log"
```

### Option 5: Create Framebuffer Device

Manually create framebuffer if possible:

```bash
# Check if display is detected
sudo modprobe fbcon
sudo modprobe fb_sys_fops

# Try to initialize framebuffer
sudo fbset -i

# Or check if display device exists
ls -la /sys/class/graphics/
```

## Current Status

- ✅ Display plugin deployed
- ✅ Console visualization working (in logs)
- ❌ Framebuffer not available (`/dev/fb0` missing)
- ❌ Display port shows black

## Quick Fix: View Console Output

For now, view the visualization via SSH:

```bash
ssh melvin@169.254.123.100
tail -f ~/melvin_system/melvin.log
```

The console output shows:
- Real-time tick counter
- Node and edge counts
- Graph statistics
- Updates every 10 ticks

## To Enable Hardware Display

1. **Connect display to Jetson HDMI/eDP port**
2. **Enable framebuffer console** (see Option 1)
3. **Reboot Jetson**
4. **Verify framebuffer exists**: `ls -l /dev/fb*`
5. **Restart Melvin**: The display plugin will auto-detect framebuffer

## Testing Display

Once framebuffer is enabled, test it:

```bash
# Simple test
echo -e "\033[2J\033[H" > /dev/tty0
echo "Display test" > /dev/tty0

# Or use framebuffer directly
sudo sh -c 'echo "Test" > /dev/fb0'
```

Once working, Melvin's display plugin will automatically use it!


