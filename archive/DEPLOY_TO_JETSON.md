# Deploy Melvin to Jetson

## Quick Start

### Option 1: Via USB (Easiest)

1. **Plug Jetson USB into Mac**
2. **Run deployment:**
   ```bash
   ./deploy_via_usb.sh
   ```
3. **Eject USB safely**
4. **On Jetson:**
   ```bash
   # Copy from USB (adjust path if needed)
   cp -r /media/*/melvin ~/melvin
   # or
   cp -r /mnt/*/melvin ~/melvin
   
   cd ~/melvin
   ./setup_jetson.sh
   ./start_melvin.sh
   ```

### Option 2: Via SSH/Network

1. **Find Jetson IP:**
   ```bash
   # On Jetson, run: hostname -I
   # Or use: ping jetson.local
   ```

2. **Deploy:**
   ```bash
   ./deploy_to_jetson.sh <jetson_ip>
   # or if hostname works:
   ./deploy_to_jetson.sh jetson.local
   ```

3. **On Jetson:**
   ```bash
   ssh melvin@<jetson_ip>
   cd ~/melvin
   ./setup_jetson.sh
   ./start_melvin.sh
   ```

## Monitor from Mac

Once Melvin is running on Jetson:

```bash
# Monitor remotely
./monitor_jetson.sh <jetson_ip>
```

This shows live stats:
- Ticks, Nodes, Edges
- Patterns discovered
- Feedback loop status

## Feed C Files to Melvin

While Melvin is running, feed it C files to digest:

```bash
# Feed single file
./feed_c_files.sh <jetson_ip> file.c

# Feed multiple files
./feed_c_files.sh <jetson_ip> file1.c file2.c file3.c

# Feed entire directory
./feed_c_files.sh <jetson_ip> --dir /path/to/c/files/
```

Files will be copied to `~/melvin/ingested_repos/` on Jetson, and Melvin's `parse_c` node will automatically parse them when it activates.

## Files Deployed

- `melvin.c` - Core runtime
- `melvin.h` - Headers
- `melvin.m` - Brain file (if exists)
- `start_melvin.sh` - Start script
- `monitor_melvin.sh` - Monitor script
- `stop_melvin.sh` - Stop script
- `setup_jetson.sh` - Setup script (created on Jetson)

## Troubleshooting

### USB Not Found
- Make sure USB is plugged in and mounted
- Check `/Volumes/` directory
- Try manually: `ls /Volumes/`

### SSH Connection Failed
- Check Jetson is on same network
- Verify IP: `ping <jetson_ip>`
- Check SSH is enabled on Jetson
- Try: `ssh melvin@<jetson_ip>` manually

### Monitor Not Working
- Make sure Melvin is running: `ssh melvin@<jetson_ip> "ps aux | grep melvin"`
- Check brain file exists: `ssh melvin@<jetson_ip> "ls -lh ~/melvin/melvin.m"`
- Try local monitor on Jetson: `ssh melvin@<jetson_ip> "cd ~/melvin && ./monitor_melvin.sh"`

## What Happens

1. **Startup**: `parse_c` node activates → parses all `.c` files in:
   - Current directory
   - `plugins/` directory  
   - `ingested_repos/` directory

2. **Parsing**: Creates nodes for:
   - Functions
   - Parameters
   - Function calls
   - Code structure

3. **Pattern Discovery**: `induce_patterns()` finds similar code patterns

4. **Learning**: Graph learns to use discovered patterns

5. **Feedback Loop**: System outputs feed back as inputs

## Example Workflow

```bash
# 1. Deploy to Jetson
./deploy_to_jetson.sh 192.168.1.100

# 2. Start on Jetson (via SSH)
ssh melvin@192.168.1.100 "cd ~/melvin && ./start_melvin.sh &"

# 3. Monitor from Mac
./monitor_jetson.sh 192.168.1.100

# 4. Feed C files while it runs
./feed_c_files.sh 192.168.1.100 my_plugin.c
./feed_c_files.sh 192.168.1.100 --dir plugins/
```

