# Enable Display Port on Jetson Orin AGX

## Problem

Display port is black - console output exists but display doesn't light up.

## Solution

The Jetson display port needs to be enabled in the boot configuration.

### Step 1: Check Current Configuration

```bash
ssh melvin@169.254.123.100
cat /boot/extlinux/extlinux.conf | grep -E 'video|console|fbcon'
```

### Step 2: Enable Display in Boot Config

Edit boot configuration:

```bash
sudo nano /boot/extlinux/extlinux.conf
```

Find the boot entry and add:

```
video=HDMI-A-1:1920x1080@60
# OR for eDP:
# video=LVDS-1:1920x1080@60
```

Also ensure console is enabled:

```
console=tty0
fbcon=map:0
```

### Step 3: Reboot

```bash
sudo reboot
```

After reboot, the display should work.

### Step 4: Verify Display

After reboot:

```bash
# Check if framebuffer exists
ls -l /dev/fb*

# Check console output
echo "TEST" | sudo tee /dev/tty1

# Switch to console 1
sudo chvt 1
```

## Alternative: View in Logs

If display can't be enabled, view output in logs:

```bash
ssh melvin@169.254.123.100 'tail -f ~/melvin_system/melvin.log'
```

## Current Status

- ✅ Display plugin: Deployed and working
- ✅ Console output: Writing to /dev/tty1
- ✅ Melvin: Running and generating visualization
- ❌ Display port: May not be enabled in boot config

## Quick Test

To test if display is connected and powered:

```bash
ssh melvin@169.254.123.100
sudo sh -c 'echo "MELVIN TEST" > /dev/tty1'
sudo chvt 1
```

If display still black after `chvt 1`, the display port needs to be enabled in boot config.


