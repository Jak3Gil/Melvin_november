# Melvin Dashboard - Production Tools

A web-based dashboard for monitoring and controlling Melvin on the Jetson.

## Features

- **Real-time Graph Monitoring**: See nodes, edges, chaos, activation in real-time
- **Drag & Drop Pattern Feeding**: Easily feed patterns to Melvin
- **3D Graph Visualization**: Interactive 3D view of nodes and edges
- **Output Display**: See Melvin's outputs and responses
- **Jetson Connection**: Connect via USB/SSH to Jetson

## Usage

### On Jetson

```bash
# Start the dashboard server
python3 tools/melvin_dashboard.py --brain /path/to/melvin_brain.m --port 8080

# Or with custom host
python3 tools/melvin_dashboard.py --brain /tmp/melvin_brain.m --host 0.0.0.0 --port 8080
```

### On Your Mac/Laptop

1. **Connect to Jetson via SSH:**
   ```bash
   ssh melvin@169.254.123.100
   # Or via USB serial
   ```

2. **Start Dashboard on Jetson:**
   ```bash
   cd ~/melvin
   python3 tools/melvin_dashboard.py --brain /tmp/melvin_brain.m
   ```

3. **Open Dashboard in Browser:**
   - Local: `http://localhost:8080`
   - Remote: `http://169.254.123.100:8080`

4. **Use the Dashboard:**
   - Drag & drop files to feed patterns
   - Watch real-time graph state
   - Explore 3D visualization
   - Monitor outputs

## Requirements

- Python 3.6+
- Web browser (Chrome, Firefox, Safari)
- Access to Jetson (SSH or USB)

## Architecture

- **Backend**: Python HTTP server (`melvin_dashboard.py`)
- **Frontend**: HTML/CSS/JavaScript with Three.js for 3D
- **Communication**: REST API + polling for real-time updates

## Future Enhancements

- WebSocket for real-time updates (instead of polling)
- Direct .m file parsing for accurate node positions
- Pattern file format support
- Export/import graph state
- Historical data visualization

