# Melvin Control System

## Philosophy: The Graph Never Stops

**Key Principle:** Melvin's brain is designed to run continuously. The graph self-regulates its activity through UEL physics:
- When active: High chaos → graph processes more
- When stable: Low chaos → graph reduces activity (but doesn't stop)
- Self-regulation: Nodes 255-259 monitor and adjust activity

**The graph should NOT stop by itself** - it's a continuous learning system.

## Control Mechanisms

### 1. Service Control (System Level)

```bash
# Start Melvin
./tools/melvin_service.sh start

# Stop Melvin (graceful shutdown)
./tools/melvin_service.sh stop

# Restart
./tools/melvin_service.sh restart

# Check status
./tools/melvin_service.sh status

# View logs
./tools/melvin_service.sh logs
```

### 2. Systemd Service (Production)

```bash
# Install service
sudo cp tools/melvin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable melvin
sudo systemctl start melvin

# Control
sudo systemctl start melvin
sudo systemctl stop melvin
sudo systemctl restart melvin
sudo systemctl status melvin
```

### 3. Control API (Programmatic)

```bash
# Start control API
python3 tools/melvin_control_api.py --port 8081

# Use API
curl http://localhost:8081/api/status
curl http://localhost:8081/api/control/start
curl http://localhost:8081/api/control/stop
curl http://localhost:8081/api/control/pause
curl http://localhost:8081/api/control/resume
```

### 4. Dashboard Integration

The dashboard can control Melvin via the Control API:
- Start/Stop buttons
- Pause/Resume for maintenance
- Status monitoring

## How It Works

### Graph Self-Regulation

The graph controls its own activity through UEL physics:

1. **High Chaos** → Graph processes more (more active)
2. **Low Chaos** → Graph reduces activity (but keeps running)
3. **Self-Regulation Nodes (255-259)**:
   - Monitor chaos levels
   - Adjust processing rate
   - Control input throttling

### External Control

**Signals:**
- `SIGTERM` / `SIGINT` → Graceful shutdown (saves state, closes cleanly)
- `SIGSTOP` → Pause (freezes graph, keeps state)
- `SIGCONT` → Resume (continues from paused state)

**The graph itself never decides to stop** - only external signals can stop it.

## Running on Jetson

### Quick Start

```bash
# 1. Build Melvin
cd ~/melvin
make melvin_hardware_runner

# 2. Start as service
./tools/melvin_service.sh start

# 3. Check it's running
./tools/melvin_service.sh status
```

### Production Setup

```bash
# 1. Install systemd service
sudo cp tools/melvin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable melvin

# 2. Start
sudo systemctl start melvin

# 3. Monitor
sudo systemctl status melvin
journalctl -u melvin -f
```

### With Dashboard

```bash
# Terminal 1: Start Melvin
./tools/melvin_service.sh start

# Terminal 2: Start Control API
python3 tools/melvin_control_api.py

# Terminal 3: Start Dashboard
python3 tools/melvin_dashboard_app.py

# Dashboard can now control Melvin via API
```

## Control Flow

```
External Control (You)
    ↓
Control API / Service Script
    ↓
Signal (SIGTERM/SIGINT/SIGSTOP/SIGCONT)
    ↓
Melvin Runner (signal handler)
    ↓
Graceful Shutdown / Pause / Resume
    ↓
Graph State Saved / Preserved
```

## Important Notes

1. **The graph never stops by itself** - it's continuous
2. **Self-regulation reduces activity** when stable, but keeps running
3. **External control is needed** to actually stop/pause
4. **Graceful shutdown** saves state to .m file
5. **Pause/Resume** preserves state without saving

## Troubleshooting

**Graph seems to stop:**
- Check if it's actually paused (low activity, not stopped)
- Check logs: `./tools/melvin_service.sh logs`
- Check status: `./tools/melvin_service.sh status`

**Can't stop Melvin:**
- Try: `kill -TERM $(cat /tmp/melvin.pid)`
- Force: `kill -KILL $(cat /tmp/melvin.pid)`

**Graph not responding:**
- May be in low-activity state (self-regulated)
- Feed some input to wake it up
- Check chaos levels in dashboard

