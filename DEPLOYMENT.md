# Melvin Deployment Guide

## Quick Deploy

### Deploy with existing brain (preserves learned patterns):
```bash
./deploy_to_jetson.sh
```

### Deploy with fresh brain (starts from scratch):
```bash
./deploy_to_jetson.sh reset_brain
```

## What Gets Deployed

**Source files copied:**
- `src/melvin.c` - Core UEL physics
- `src/melvin.h` - Header definitions
- `src/host_syscalls.c` - Syscall implementations
- `src/melvin_tools.c` - Tool integrations
- `src/melvin_tools.h` - Tool headers
- `src/melvin_hardware_*.c` - Hardware I/O
- `src/melvin_hardware_runner.c` - Main runner

**What happens:**
1. Stops running Melvin instance
2. Backs up current `brain.m` (timestamped)
3. Copies source files to Jetson
4. Rebuilds binary on Jetson
5. Starts Melvin with updated code

## When to Reset brain.m

### Keep brain.m (default):
- **Code changes that don't affect graph structure:**
  - Bug fixes
  - Performance improvements
  - NaN protection
  - Dynamic thresholds
  - Tool improvements
  - Hardware fixes

### Reset brain.m (use `reset_brain` flag):
- **Code changes that affect graph structure:**
  - New node/edge fields added
  - Header format changes
  - UEL physics major changes
  - File format version changes

## Current Status

**Brain location:** `/mnt/melvin_ssd/melvin_brain/brain.m` (4TB SSD)

**Backup location:** `/mnt/melvin_ssd/melvin_brain/brain.m.backup.*`

**Logs:** `/mnt/melvin_ssd/melvin_brain/melvin.log`

## Monitoring

```bash
# Watch logs in real-time
ssh melvin@169.254.123.100 'tail -f /mnt/melvin_ssd/melvin_brain/melvin.log'

# Check status
ssh melvin@169.254.123.100 'ps aux | grep melvin_hardware_runner'

# Check brain size
ssh melvin@169.254.123.100 'ls -lh /mnt/melvin_ssd/melvin_brain/brain.m'
```

## Manual Deployment

If the script doesn't work, manually:

```bash
# 1. Copy files
sshpass -p '123456' scp -o StrictHostKeyChecking=no \
    src/melvin.c src/melvin.h src/*.c \
    melvin@169.254.123.100:~/melvin/src/

# 2. Rebuild on Jetson
sshpass -p '123456' ssh -o StrictHostKeyChecking=no melvin@169.254.123.100 \
    'cd ~/melvin && make melvin_hardware_runner'

# 3. Restart
sshpass -p '123456' ssh -o StrictHostKeyChecking=no melvin@169.254.123.100 \
    'cd ~/melvin && killall melvin_hardware_runner; \
     nohup ./melvin_hardware_runner /mnt/melvin_ssd/melvin_brain/brain.m default default > /mnt/melvin_ssd/melvin_brain/melvin.log 2>&1 &'
```

