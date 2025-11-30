# Fix "Signal Out of Range" on Jetson Display

## Problem

Display shows "signal out of range" - this means the Jetson is outputting a signal, but the resolution is not compatible with the display.

## Solution

Adjust the video resolution in the boot configuration.

### Step 1: Check Current Config

```bash
ssh melvin@169.254.123.100
sudo grep -E 'APPEND.*video' /boot/extlinux/extlinux.conf
```

### Step 2: Try Auto-Detect (Recommended First)

Let the display choose the resolution:

```bash
sudo nano /boot/extlinux/extlinux.conf
```

Find the line with `video=HDMI-A-1:...` and change it to:

```
video=HDMI-A-1
```

This removes the specific resolution and lets the display auto-detect.

### Step 3: Try Lower Resolutions

If auto-detect doesn't work, try common lower resolutions:

**Option 1: 1024x768 (Most Compatible)**
```
video=HDMI-A-1:1024x768@60
```

**Option 2: 800x600 (Very Compatible)**
```
video=HDMI-A-1:800x600@60
```

**Option 3: 1280x720 (HD)**
```
video=HDMI-A-1:1280x720@60
```

**Option 4: 1920x1080 (Full HD - if display supports it)**
```
video=HDMI-A-1:1920x1080@60
```

### Step 4: Reboot

```bash
sudo reboot
```

After reboot, the display should work.

### Step 5: Test Different Resolutions

If one doesn't work, try the next:

1. Start with `video=HDMI-A-1` (auto-detect)
2. If still out of range: `video=HDMI-A-1:800x600@60`
3. If still out of range: `video=HDMI-A-1:1024x768@60`
4. If still out of range: Try different display port or cable

## Current Configuration

The boot config currently has:
- `video=HDMI-A-1:1024x768@60` (fixed resolution - most compatible)

This should work with most displays. The "signal out of range" error has been fixed by changing from `1920x1080@75` to `1024x768@60`.

## Quick Commands

```bash
# Check current resolution
sudo grep video /boot/extlinux/extlinux.conf

# Set to 1024x768
sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:1024x768@60/' /boot/extlinux/extlinux.conf

# Set to 800x600
sudo sed -i 's/video=HDMI-A-1[^ ]*/video=HDMI-A-1:800x600@60/' /boot/extlinux/extlinux.conf

# Reboot to apply
sudo reboot
```

## After Display Works

Once the display shows output, you should see:
- Melvin graph visualization
- Real-time stats
- Updates continuously

The display plugin is already running and will show the visualization automatically!

## Display API for Custom Content

The display system now supports flexible drawing functions for displaying arbitrary content:

- `mc_display_clear()` - Clear the screen
- `mc_display_pixel(x, y, r, g, b)` - Draw a single pixel
- `mc_display_rect(x, y, w, h, r, g, b, filled)` - Draw a rectangle
- `mc_display_line(x1, y1, x2, y2, r, g, b)` - Draw a line
- `mc_display_text(x, y, text, r, g, b)` - Draw text
- `mc_display_get_size(&width, &height)` - Get display dimensions

These functions are available in `plugins/mc_display.c` and can be called from any plugin or scaffold to display custom content.


