# Melvin - Simple Start Guide

## Quick Start

### 1. Start Melvin
```bash
./start_melvin.sh
```

This will:
- Start melvin.m running continuously
- Run in the background
- Log to `melvin.log`

### 2. Monitor Live Stats
In another terminal:
```bash
./monitor_melvin.sh
```

This shows live stats:
- **Ticks**: Current tick count and rate
- **Nodes**: Total, capacity, usage, types
- **Edges**: Total, capacity, usage
- **Patterns**: Number of discovered patterns
- **Feedback Loop**: FB_IN/FB_OUT status
- **Parameters**: Current system parameters

### 3. Stop Melvin
```bash
./stop_melvin.sh
```

Or press Ctrl+C in the start terminal.

## Files

- `start_melvin.sh` - Start button (runs melvin.m)
- `monitor_melvin.sh` - Live monitor (shows nodes, edges, ticks)
- `stop_melvin.sh` - Stop button
- `melvin.m` - The brain file (memory-mapped graph)

## Monitor Options

```bash
# Default refresh (0.5 seconds)
./monitor_melvin.sh

# Custom refresh rate
./monitor_melvin.sh melvin.m 1.0

# Different brain file
./monitor_melvin.sh custom_brain.m
```

## What You'll See

The monitor displays:
- **Ticks**: How many cycles completed
- **Nodes**: Graph nodes (Data, Control, MC, Patterns)
- **Edges**: Connections between nodes
- **Patterns**: Discovered code patterns
- **Feedback Loop**: Internal energy flow (FB_IN → FB_OUT)

## Notes

- Melvin runs continuously until stopped
- The monitor reads the brain file directly (non-blocking)
- You can run multiple monitors on the same brain file
- Stats update in real-time as melvin processes

