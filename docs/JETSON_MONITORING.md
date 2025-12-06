# Monitoring and Feeding Melvin from Mac

## Quick Start

### 1. Monitor Melvin Live
```bash
./monitor_melvin.sh
```
This connects to the Jetson and shows live stats every 5 seconds:
- Node/edge counts
- Average chaos, activation, edge strength
- Drive mechanism states
- Active nodes percentage
- Queue status

Press Ctrl+C to stop.

### 2. Feed New Files to Melvin
```bash
# Feed a C file
./feed_melvin.sh hello.c

# Feed with custom port node and energy
./feed_melvin.sh new_code.c 256 0.2

# Feed any file type
./feed_melvin.sh data.txt
```

## What Gets Fed

All files are fed as **raw bytes** to the graph:
- **C source files** → Melvin learns code patterns
- **Text files** → Melvin learns language patterns  
- **Binary data** → Melvin learns byte patterns
- **Any file** → Becomes part of the energy landscape

## How It Works

1. **File Transfer**: File is copied to Jetson
2. **Byte Feeding**: Each byte is fed to the graph via `melvin_feed_byte()`
3. **UEL Propagation**: Graph processes the new data through UEL physics
4. **Learning**: Graph learns patterns, reduces chaos
5. **Persistence**: Changes are synced to `brain.m` file

## Direct Jetson Commands

You can also SSH directly and run commands:

```bash
# Monitor
sshpass -p '123456' ssh melvin@169.254.123.100 \
  "cd /mnt/melvin_ssd/melvin && ./melvin_monitor brain.m 5"

# Feed file
sshpass -p '123456' ssh melvin@169.254.123.100 \
  "cd /mnt/melvin_ssd/melvin && ./melvin_feed_file brain.m /path/to/file.c 0 0.1"
```

## Continuous Learning

Melvin can run continuously and learn from:
1. **Pre-loaded corpus** (in cold_data region)
2. **New files you feed** (via feed_melvin.sh)
3. **Internal curiosity** (when chaos is low, it seeks new data)
4. **Self-directed reading** (graph can copy from cold_data to blob)

## Tools Available

- `melvin_monitor` - Live stats monitoring
- `melvin_feed_file` - Feed files to graph
- `melvin_pack_corpus` - Pack corpus into cold_data
- `melvin_seed_instincts` - Seed bootstrap patterns

All tools are on Jetson at: `/mnt/melvin_ssd/melvin/`

