# Jetson Monitoring Guide

Always know what's happening on your Jetson when connected via USB.

## Quick Start

### Option 1: Interactive Terminal (Recommended for commands)
```bash
./jetson_terminal.sh
```
Opens a live SSH session to the Jetson. You can run commands directly and stay connected.

### Option 2: Live Dashboard (Recommended for monitoring)
```bash
./jetson_live_monitor.sh
```
Auto-refreshing dashboard that updates every 3 seconds. Shows:
- Connection status
- Running processes with CPU/memory usage
- Latest log activity
- Brain file status
- System resources

### Option 3: One-time Status Check
```bash
./check_melvin_status.sh
```
Quick snapshot of current Jetson state.

## Use Cases

### 🎯 I want to watch logs in real-time
```bash
./jetson_terminal.sh
# Once connected:
tail -f /tmp/melvin_run.log
```

### 🎯 I want to see if Melvin is running
```bash
./jetson_live_monitor.sh
# Leave it running in a terminal window
```

### 🎯 I want to run commands while Melvin is running
```bash
./jetson_terminal.sh
# Once connected, you can:
ps aux | grep melvin              # Check processes
./melvin_monitor brain.m 5        # Monitor brain stats
top                               # Watch system resources
```

### 🎯 I want a quick status check
```bash
./check_melvin_status.sh
```

## Best Practices

### Keep a Monitor Running
When you're working with Melvin on the Jetson, keep `jetson_live_monitor.sh` running in a separate terminal window. This gives you constant visibility into what's happening.

### Use Two Terminals
**Terminal 1:** Run `jetson_live_monitor.sh` for passive monitoring
**Terminal 2:** Run `jetson_terminal.sh` when you need to execute commands

### Check Connection First
All scripts will show if the Jetson is reachable. If you see:
```
❌ Cannot reach Jetson at 169.254.123.100
```
Check your USB connection.

## Troubleshooting

### "Cannot reach Jetson"
1. Check USB cable is connected
2. Check Jetson is powered on
3. Verify USB network interface is up:
   ```bash
   ifconfig | grep 169.254
   ```

### "Permission denied"
The scripts use the credentials:
- IP: 169.254.123.100
- User: melvin
- Password: 123456

If these change, update them in the scripts.

### Scripts hang or timeout
The Jetson might be busy or unresponsive. Try:
1. Ctrl+C to cancel
2. Wait a moment
3. Try again

## Connection Details

All scripts connect via USB using:
- **IP Address:** 169.254.123.100 (USB network)
- **Protocol:** SSH over USB
- **Working Directory:** /home/melvin/melvin

The USB connection provides a direct network link to the Jetson, independent of WiFi/Ethernet.

## What Each Script Shows

### jetson_terminal.sh
- Interactive shell on Jetson
- Full command access
- Stays connected until you type `exit`
- Best for: Running commands, debugging, manual inspection

### jetson_live_monitor.sh
- Auto-refreshing every 3s
- Non-interactive (watch mode)
- Shows: processes, logs, brain file, resources
- Best for: Passive monitoring, knowing what's happening at a glance

### check_melvin_status.sh
- One-time snapshot
- Quick check
- Best for: Quick status verification

## Advanced: Custom Monitoring

You can modify `jetson_live_monitor.sh` to add custom checks:

```bash
# Add after line 50 (in the while loop):
echo "CUSTOM CHECK:"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "YOUR_CUSTOM_COMMAND_HERE"
```

Common additions:
- GPU usage: `tegrastats`
- Disk space: `df -h`
- Network traffic: `ifconfig`
- Temperature: `cat /sys/class/thermal/thermal_zone*/temp`

