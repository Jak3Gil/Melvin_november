# Jetson Orin AGX Display Fix - Based on Online Research

## Problem
"Signal Out of Range" error on Jetson Orin AGX display

## Root Causes (from NVIDIA documentation and forums)

1. **Resolution/Refresh Rate Mismatch** - Monitor doesn't support the set resolution/refresh rate
2. **EDID Issues** - Monitor capabilities not detected correctly
3. **Adapter Compatibility** - Some DP-to-HDMI converters don't work with Jetson Orin AGX
4. **Hotplug Issues** - Display should be connected before boot
5. **Known Bugs** - Some resolutions cause display to go blank (e.g., 4K@60Hz on certain monitors)

## Solutions to Try (in order)

### Solution 1: Let EDID Auto-Negotiate (Recommended First)
Remove the resolution parameter entirely and let the display negotiate:

```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Solution 2: Try Standard HD Resolution (1280x720@60)
Most monitors support this:

```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:1280x720@60/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Solution 3: Try DisplayPort Instead of HDMI
If using HDMI adapter, try DisplayPort directly:

```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1/video=DP-0/' /boot/extlinux/extlinux.conf && sudo reboot"
```

Or with resolution:
```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=DP-0:1280x720@60/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Solution 4: Try Lower Refresh Rate
Some monitors don't support 60Hz at certain resolutions:

```bash
# Try 50Hz instead
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/@60/@50/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Solution 5: Try Very Safe Resolution (640x480@60)
VGA mode - almost all monitors support this:

```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:640x480@60/' /boot/extlinux/extlinux.conf && sudo reboot"
```

## Hardware Checks

1. **Check HDMI Cable** - Try a different, high-quality HDMI cable
2. **Check Adapter** - If using DP-to-HDMI adapter, try a different one (NVIDIA notes some don't work)
3. **Check Monitor** - Try a different monitor/TV to rule out monitor issues
4. **Connect Before Boot** - Connect display before powering on (hotplugging can cause issues)

## Check Current Settings

```bash
# Check boot config
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo grep video /boot/extlinux/extlinux.conf | head -1"

# Check kernel cmdline (what's actually running)
sshpass -p '123456' ssh melvin@169.254.123.100 "cat /proc/cmdline | grep -o 'video=[^ ]*'"

# Check framebuffer
sshpass -p '123456' ssh melvin@169.254.123.100 "cat /sys/class/graphics/fb0/virtual_size"
```

## Known Issues from NVIDIA Documentation

1. **DP-to-HDMI Adapters** - Some CableCreation branded adapters don't work
2. **4K@60Hz** - Doesn't work on some monitors (ACER Predator X27)
3. **Display Blank During Boot** - Known intermittent issue on Jetson AGX Orin
4. **Hotplugging** - Can cause corrupted screen - connect before boot

## Recommended Resolution Order to Try

1. **Auto-detect** (`video=HDMI-A-1`) - Let EDID negotiate
2. **1280x720@60** - Standard HD, widely supported
3. **1024x768@60** - XGA, very common
4. **800x600@60** - SVGA, very compatible
5. **640x480@60** - VGA, universal support

## If Still Not Working

1. Check monitor manual for supported resolutions/refresh rates
2. Try different HDMI port on monitor
3. Try DisplayPort output instead of HDMI
4. Update Jetson software: `sudo apt update && sudo apt upgrade`
5. Check NVIDIA forums for your specific monitor model

## Current Configuration

As of latest change:
- **Boot config**: `video=HDMI-A-1:1280x720@60`
- **Status**: Needs reboot to apply

