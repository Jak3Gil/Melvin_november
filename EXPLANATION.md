# What You're Seeing - Melvin Status Explanation

## The Process List (ps aux | grep melvin)

When you see multiple `melvin_run_continuous` processes, here's what they mean:

### Process Types:
1. **`/mnt/melvin_ssd/melvin/melvin_run_continuous`** - Running from the SSD mount point
2. **`./melvin_run_continuous`** - Running from current directory (usually ~/melvin)

### Why Multiple Processes?
- Each time you run the script, it starts a new process
- Old processes might still be running in the background
- Multiple instances can cause conflicts (trying to write to the same brain file)

## What Melvin is Actually Doing

From the log output, you can see:
- **Iteration**: How many UEL physics cycles have run (e.g., 2460+)
- **Nodes**: Number of nodes in the graph (10,000)
- **Edges**: Number of connections (177)
- **Chaos**: Measure of system incoherence (0.229)
- **Activation**: Average node activation level (0.285)
- **Synced to disk**: Brain file saved every 60 seconds

## Status Indicators

✅ **Good Signs:**
- Numbers are stable (not crashing)
- "Synced to disk" appears regularly
- Process has been running for a while

⚠️ **Warning Signs:**
- Multiple processes running simultaneously
- Numbers not changing (might be stuck)
- "Failed to open" errors

## How to Check Status

Run: `./check_melvin_status.sh`

This will show you:
- How many processes are running
- Latest activity from the log
- Brain file locations and sizes

## How to Clean Up

To stop all melvin processes:
```bash
sshpass -p "123456" ssh melvin@169.254.123.100 "pkill -f melvin_run_continuous"
```

To start fresh (one instance):
```bash
./run_melvin_jetson_usb.sh
```

