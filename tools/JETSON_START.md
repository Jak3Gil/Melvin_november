# Starting Melvin on Jetson

## Quick Start

```bash
# 1. Build Melvin
cd ~/melvin
make melvin_hardware_runner

# 2. Start Melvin (runs continuously)
./tools/melvin_service.sh start

# 3. Check it's running
./tools/melvin_service.sh status

# 4. View logs
./tools/melvin_service.sh logs
```

## How It Works

### The Graph Never Stops

**Key Principle:** Melvin's brain runs continuously. The graph self-regulates:
- **High activity** → More processing (high chaos)
- **Low activity** → Less processing (low chaos, but still running)
- **Self-regulation nodes (255-259)** control activity automatically

**The graph does NOT stop by itself** - it's a continuous learning system.

### Control Methods

**1. Service Script (Easiest)**
```bash
./tools/melvin_service.sh start   # Start
./tools/melvin_service.sh stop    # Stop (graceful)
./tools/melvin_service.sh status   # Check status
./tools/melvin_service.sh logs    # View logs
```

**2. Systemd Service (Production)**
```bash
sudo cp tools/melvin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable melvin
sudo systemctl start melvin
sudo systemctl status melvin
```

**3. Control API (Programmatic)**
```bash
# Start API
python3 tools/melvin_control_api.py

# Control via API
curl http://localhost:8081/api/control/start
curl http://localhost:8081/api/control/stop
curl http://localhost:8081/api/status
```

**4. Dashboard (GUI)**
```bash
# Start dashboard (has control buttons)
python3 tools/melvin_dashboard_app.py
```

## What Happens When You Start

1. **Melvin opens the .m file** (creates if new)
2. **Soft structure initializes** (creates nodes/edges for patterns)
3. **Hardware initializes** (mic, speaker, cameras)
4. **Main loop starts** (runs continuously):
   - Hardware threads feed data to graph
   - Graph processes via UEL physics
   - Graph self-regulates activity
   - State syncs to disk every 60 seconds
5. **Runs forever** until you stop it

## Stopping Melvin

**Graceful Shutdown:**
```bash
./tools/melvin_service.sh stop
# Or: kill -TERM $(cat /tmp/melvin.pid)
```

This:
- Saves graph state to .m file
- Closes hardware cleanly
- Exits gracefully

**Force Stop (if needed):**
```bash
kill -KILL $(cat /tmp/melvin.pid)
```

## Monitoring

**Check if running:**
```bash
./tools/melvin_service.sh status
```

**View real-time logs:**
```bash
./tools/melvin_service.sh logs
```

**Use dashboard:**
```bash
python3 tools/melvin_dashboard_app.py
# Open http://localhost:8080
```

## Self-Regulation

The graph controls its own activity:

- **Nodes 255-259**: Monitor chaos and adjust activity
- **High chaos** → More processing (graph is learning)
- **Low chaos** → Less processing (graph is stable)
- **Graph never stops** - only reduces activity when stable

**This is normal!** Low activity doesn't mean it stopped - it's self-regulating.

## Troubleshooting

**"Graph seems stopped":**
- Check if it's actually paused (low activity, not stopped)
- Feed some input to wake it up
- Check logs: `./tools/melvin_service.sh logs`

**"Can't start":**
- Check brain file path exists or is writable
- Check hardware devices are available
- Check logs for errors

**"Graph not learning":**
- Feed it data (audio, video, files)
- Check chaos levels in dashboard
- Graph learns from patterns, needs input

## Production Setup

```bash
# 1. Install as systemd service
sudo cp tools/melvin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable melvin

# 2. Start
sudo systemctl start melvin

# 3. Auto-starts on boot
# Monitor with:
journalctl -u melvin -f
```

