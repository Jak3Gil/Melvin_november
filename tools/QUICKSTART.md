# Melvin Dashboard - Quick Start

## Start Dashboard on Jetson

```bash
# SSH to Jetson
ssh melvin@169.254.123.100

# Navigate to melvin directory
cd ~/melvin

# Start dashboard
python3 tools/melvin_dashboard.py --brain /tmp/melvin_brain.m --port 8080

# Or use the script
./tools/start_dashboard.sh /tmp/melvin_brain.m 8080
```

## Access from Your Mac

1. **Open browser:**
   ```
   http://169.254.123.100:8080
   ```

2. **Or if running locally:**
   ```
   http://localhost:8080
   ```

## Features

### Real-time Monitoring
- See graph state (nodes, edges, chaos, activation)
- Updates every 500ms
- Connection status indicator

### Feed Patterns
- Drag & drop files onto the drop zone
- Files are automatically fed to the graph
- See file list and status

### 3D Visualization
- Interactive 3D view of nodes and edges
- Click and drag to rotate
- Scroll to zoom
- Adjust node size with slider

### Output Display
- See Melvin's outputs in real-time
- Color-coded messages (info, success, error)
- Scrollable log

## Troubleshooting

**Dashboard won't start:**
- Check Python 3 is installed: `python3 --version`
- Check port is available: `netstat -an | grep 8080`

**Can't connect from Mac:**
- Check Jetson IP: `ip addr show`
- Check firewall: `sudo ufw allow 8080`
- Try localhost on Jetson first: `http://localhost:8080`

**3D visualization not working:**
- Check browser console for errors
- Try different browser (Chrome recommended)
- Check Three.js is loading

## Next Steps

- Connect to running Melvin instance
- Feed conversation data
- Feed C files for motor control
- Monitor graph growth in real-time

