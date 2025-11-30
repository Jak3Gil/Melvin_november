# Launch Melvin Dashboard as App

## Quick Launch (Python - Opens Browser Window)

```bash
# Simplest way - opens in separate browser window
python3 tools/melvin_dashboard_app.py

# With custom brain file
python3 tools/melvin_dashboard_app.py --brain /path/to/brain.m

# Or use the launcher script
./tools/launch_dashboard.sh
```

## Electron App (True Desktop App)

```bash
# First time: Install Electron
cd tools && ./install_electron.sh

# Then launch
cd tools && npm start

# Or use the launcher
./tools/launch_dashboard.sh
```

## What Happens

1. **Python App Mode:**
   - Starts dashboard server
   - Automatically opens browser in NEW WINDOW
   - Dashboard runs in that window
   - Close window or Ctrl+C to stop

2. **Electron App Mode:**
   - Creates native desktop app window
   - No browser needed
   - Better OS integration
   - Can be packaged as .app/.exe

## Features

✅ Separate window (not a browser tab)
✅ Real-time graph monitoring
✅ Drag & drop pattern feeding
✅ 3D visualization
✅ Output display

