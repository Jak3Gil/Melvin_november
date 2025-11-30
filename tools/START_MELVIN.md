# How to Start Melvin on Jetson

## The Graph Never Stops (By Design)

**Key Principle:** Melvin's brain is continuous. The graph self-regulates:
- When active: High chaos → processes more
- When stable: Low chaos → reduces activity (but keeps running)
- Self-regulation: Nodes 255-259 control activity automatically

**The graph does NOT stop by itself** - only external control can stop it.

## Quick Start

```bash
# On Jetson
cd ~/melvin

# Build (if needed)
make melvin_hardware_runner

# Start Melvin (runs continuously)
./tools/melvin_service.sh start

# Check status
./tools/melvin_service.sh status

# View logs
./tools/melvin_service.sh logs
```

## Control Methods

1. **Service Script** (easiest)
2. **Systemd Service** (production)
3. **Control API** (programmatic)
4. **Dashboard** (GUI with buttons)

See `tools/CONTROL_SYSTEM.md` for details.

## What Happens

1. Opens .m file (creates if new)
2. Initializes soft structure (creates nodes/edges)
3. Starts hardware (mic, speaker, cameras)
4. Runs continuously (self-regulating)
5. Syncs to disk every 60 seconds
6. Runs until you stop it

## Stopping

```bash
./tools/melvin_service.sh stop  # Graceful shutdown
```

Saves state and exits cleanly.
