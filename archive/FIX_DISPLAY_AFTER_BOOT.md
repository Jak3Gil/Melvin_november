# Fix Display "Signal Out of Range" After Boot

## Problem Identified

The display shows the startup window correctly, but then shows "signal out of range" after boot completes. This means:

1. ✅ Boot resolution is working (startup window visible)
2. ❌ Something changes the resolution after boot, causing the error

## Root Cause

Found `melvin-resolution.service` that runs after boot and tries to change the framebuffer resolution using `fbset`. This service was:
- Setting framebuffer to `1280x720` 
- But boot parameter might be different
- `fbset` may not work on modern Jetson systems using DRM/KMS

## Solution Applied

### Option 1: Disable the Service (Recommended)
The boot parameter should handle the resolution, so the service isn't needed:

```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo systemctl disable melvin-resolution.service"
```

### Option 2: Match Service to Boot Parameter
Updated both to use the same resolution (`1280x720@60`):

**Boot config:**
```
video=HDMI-A-1:1280x720@60
```

**Service:**
```
ExecStart=/bin/fbset -g 1280 720 1280 720 32
```

## Current Status

- Boot parameter: `video=HDMI-A-1:1280x720@60` ✅
- Service: Updated to `1280x720` ✅
- Service: **DISABLED** (recommended - let boot parameter handle it)

## Next Steps

1. **Reboot** to test:
   ```bash
   sshpass -p '123456' ssh melvin@169.254.123.100 "sudo reboot"
   ```

2. **If still "signal out of range":**
   - Try auto-detect (remove resolution from boot parameter):
     ```bash
     sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1/' /boot/extlinux/extlinux.conf && sudo reboot"
     ```
   
   - Try a different resolution that matches your monitor:
     ```bash
     # Try 1024x768@60
     sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:1024x768@60/' /boot/extlinux/extlinux.conf && sudo reboot"
     ```

3. **Check monitor specifications** - What resolutions does your monitor support?

## Why This Happens

On Jetson Orin AGX:
- Boot parameter sets the **HDMI output resolution** (what the monitor sees)
- Framebuffer (`/dev/fb0`) might be separate and controlled by DRM/KMS
- `fbset` may not work on modern systems
- The mismatch between HDMI output and framebuffer causes "signal out of range"

## Alternative: Use DRM/KMS Tools

If you need to change resolution at runtime, use DRM tools instead of fbset:

```bash
# Check available modes
sudo modetest -c

# Set resolution (if modetest is available)
sudo modetest -s <connector>@<mode>
```

But for most cases, the boot parameter should be sufficient.

