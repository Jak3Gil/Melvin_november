# Display Hardware Troubleshooting

## Current Status
- Boot parameter: `video=HDMI-A-1:640x480@60` (VGA - most compatible)
- Display still showing "signal out of range"

## If 640x480@60 Doesn't Work

This suggests a **hardware compatibility issue** rather than a software configuration problem.

## Questions to Answer

1. **What monitor/TV are you using?**
   - Brand and model?
   - What resolutions does it support? (check manual)
   - Does it have multiple HDMI inputs?

2. **What cable are you using?**
   - HDMI cable?
   - DisplayPort to HDMI adapter?
   - Cable length?

3. **What happens exactly?**
   - Do you see the startup window? (YES - you said this works)
   - When does "signal out of range" appear? (After boot completes?)
   - Does it flash or stay black?

4. **Have you tried:**
   - Different HDMI cable?
   - Different monitor/TV?
   - Different HDMI port on the monitor?
   - Direct connection (no adapters)?

## Next Steps to Try

### Option 1: Remove Video Parameter Entirely
Let the kernel use defaults:

```bash
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/ video=HDMI-A-1[^ ]*//' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Option 2: Try Lower Refresh Rate
Some monitors don't support 60Hz at low resolutions:

```bash
# Try 50Hz
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:640x480@50/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Option 3: Try Different HDMI Port
Jetson Orin AGX might have multiple HDMI outputs:

```bash
# Try HDMI-A-2
sshpass -p '123456' ssh melvin@169.254.123.100 "sudo sed -i 's/HDMI-A-1/HDMI-A-2/' /boot/extlinux/extlinux.conf && sudo reboot"
```

### Option 4: Check Monitor Specifications
- What is the **minimum resolution** your monitor supports?
- What is the **maximum refresh rate** at low resolutions?
- Does it support **640x480** at all?

## Hardware Issues to Consider

1. **HDMI Cable Quality**
   - Old or damaged cables can cause signal issues
   - Try a high-quality, short HDMI cable

2. **Adapter Issues**
   - If using DP-to-HDMI adapter, some don't work with Jetson
   - Try direct HDMI connection if possible

3. **Monitor EDID**
   - Monitor might not be sending proper EDID data
   - Try a different monitor to test

4. **HDMI Port on Jetson**
   - Try the other HDMI port if available
   - Check for physical damage

## Alternative: Use Serial Console

If display won't work, you can always use:
- Serial console via USB (COM8)
- SSH (already working)
- VNC or remote desktop

The display is nice-to-have but not required for operation.

## What We've Tried

✅ 1920x1080@75 → Signal out of range  
✅ 1280x720@60 → Signal out of range  
✅ 1024x768@60 → Signal out of range  
✅ 800x600@60 → Signal out of range  
✅ 640x480@60 → Signal out of range  
✅ Auto-detect (EDID) → Signal out of range  

This strongly suggests a **hardware compatibility issue** with the monitor, cable, or adapter.

