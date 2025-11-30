# Fix "No Signal" on Jetson Display

## Problem

Display shows "no signal" even after enabling HDMI in boot config.

## Solution

Set a specific lower resolution that the display supports.

### Step 1: SSH to Jetson

```bash
ssh melvin@169.254.123.100
```

### Step 2: Set Lower Resolution

Try these resolutions in order:

#### Option 1: 1024x768 (Most Compatible)

```bash
sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:1024x768@60/' /boot/extlinux/extlinux.conf
```

#### Option 2: 800x600 (Very Compatible)

```bash
sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:800x600@60/' /boot/extlinux/extlinux.conf
```

#### Option 3: 1280x720 (HD)

```bash
sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:1280x720@60/' /boot/extlinux/extlinux.conf
```

### Step 3: Verify Config

```bash
sudo grep video /boot/extlinux/extlinux.conf
```

Should show something like:
```
video=HDMI-A-1:1024x768@60
```

### Step 4: Reboot

```bash
sudo reboot
```

### Step 5: Test

After reboot:
1. Display should light up
2. Should show console output
3. Melvin visualization should appear

### If Still No Signal

Try different display port:

```bash
# Try HDMI-A-2 instead
sudo sed -i 's/video=HDMI-A-1/video=HDMI-A-2/' /boot/extlinux/extlinux.conf
sudo reboot
```

Or check if display is detected:

```bash
# After reboot, check display detection
dmesg | grep -i hdmi
dmesg | grep -i display
```

### Quick Commands

```bash
# Set to 1024x768
sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:1024x768@60/' /boot/extlinux/extlinux.conf

# Set to 800x600
sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:800x600@60/' /boot/extlinux/extlinux.conf

# Check current config
sudo grep video /boot/extlinux/extlinux.conf

# Reboot
sudo reboot
```

## Current Status

The boot config currently has:
- `video=HDMI-A-1:1024x768@60`

After reboot, this should work!

## Troubleshooting

1. **Still no signal?** Try 800x600
2. **Display flickers?** Try different refresh rate (50Hz instead of 60Hz)
3. **Wrong aspect ratio?** Try 1280x720 or 1920x1080
4. **No signal at all?** Check HDMI cable, display power, different display port

## After Display Works

Once the display shows output, Melvin's visualization will automatically appear:
- Graph stats
- Node/edge counts
- Real-time updates

The display plugin is already running!


