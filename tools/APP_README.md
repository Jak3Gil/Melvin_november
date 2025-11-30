# Melvin Dashboard - Desktop App

The dashboard can run as a standalone desktop app that opens in its own window.

## Quick Start

### Option 1: Python App (Simplest)

```bash
# Just run the Python launcher - it opens a browser window automatically
python3 tools/melvin_dashboard_app.py --brain /tmp/melvin_brain.m
```

This will:
- Start the dashboard server
- Automatically open your browser in a new window
- Keep the window separate from other browser tabs

### Option 2: Electron App (True Desktop App)

For a true desktop app experience:

```bash
# 1. Install Electron (one-time setup)
cd tools
./install_electron.sh

# 2. Launch the app
./launch_dashboard.sh /tmp/melvin_brain.m
```

Or manually:
```bash
cd tools
npm install
npm start
```

## Features

### Python App Mode
- ✅ Opens in separate browser window
- ✅ No installation needed
- ✅ Works on any platform
- ✅ Lightweight

### Electron App Mode
- ✅ True desktop app (not a browser)
- ✅ Native window controls
- ✅ Can be packaged as .app/.exe
- ✅ Better integration with OS

## Usage

### Basic Launch
```bash
# Python app (opens browser window)
python3 tools/melvin_dashboard_app.py

# Electron app
cd tools && npm start
```

### With Custom Brain File
```bash
python3 tools/melvin_dashboard_app.py --brain /path/to/brain.m
```

### With Custom Port
```bash
python3 tools/melvin_dashboard_app.py --port 9000
```

### Don't Auto-Open Browser
```bash
python3 tools/melvin_dashboard_app.py --no-browser
```

## Building Standalone App (Electron)

To create a distributable app:

```bash
cd tools
npm install
npm run build
```

This creates:
- **Mac**: `dist/Melvin Dashboard.dmg`
- **Linux**: `dist/Melvin Dashboard.AppImage`
- **Windows**: `dist/Melvin Dashboard Setup.exe`

## Troubleshooting

**Browser doesn't open:**
- Try `--no-browser` and manually open `http://localhost:8080`
- Check if port 8080 is available

**Electron app won't start:**
- Make sure Node.js is installed: `node --version`
- Run `npm install` in the tools directory
- Check Python 3 is available: `python3 --version`

**Dashboard shows "Disconnected":**
- Make sure the brain file exists
- Check the brain file path is correct
- Verify Melvin is running and creating the brain file

## Platform Notes

### Mac
- Electron app integrates with macOS (dock, menu bar)
- Python app opens in default browser

### Linux
- Both modes work
- Electron app can be packaged as AppImage

### Windows
- Both modes work
- Electron app can be packaged as .exe installer

