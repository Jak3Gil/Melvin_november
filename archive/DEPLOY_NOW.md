# Deploy Melvin to Jetson - Quick Guide

## Jetson is Connected via USB

Your Jetson is detected at: `/dev/tty.usbmodem14217250286373`

## Option 1: Get IP and Deploy via SSH

**On Jetson (if you have access to its screen/terminal):**
```bash
hostname -I
```

**Then on Mac:**
```bash
./deploy_manual.sh <JETSON_IP>
```

## Option 2: Manual Copy via USB

If SSH isn't working, you can manually copy files:

1. **On Mac - Prepare files:**
   ```bash
   mkdir -p /tmp/melvin_deploy
   cp melvin.c melvin.h melvin.m /tmp/melvin_deploy/
   cp start_melvin.sh monitor_melvin.sh stop_melvin.sh /tmp/melvin_deploy/
   cp monitor_melvin.c init_melvin_simple.c /tmp/melvin_deploy/ 2>/dev/null
   ```

2. **Copy to USB drive** (if Jetson USB mounts as storage):
   - Plug in a USB drive
   - Copy `/tmp/melvin_deploy/*` to USB
   - Eject and plug into Jetson

3. **On Jetson:**
   ```bash
   # Copy from USB
   cp -r /media/*/melvin_deploy ~/melvin
   # or
   cp -r /mnt/*/melvin_deploy ~/melvin
   
   cd ~/melvin
   gcc -o melvin melvin.c -lm -ldl
   gcc -o monitor_melvin monitor_melvin.c -lm
   chmod +x *.sh
   ./start_melvin.sh
   ```

## Option 3: Enable SSH on Jetson First

If Jetson doesn't have SSH enabled:

**On Jetson:**
```bash
sudo systemctl enable ssh
sudo systemctl start ssh
sudo systemctl status ssh
hostname -I  # Get IP address
```

**Then on Mac:**
```bash
./deploy_manual.sh <JETSON_IP>
```

## Quick Test Connection

Try connecting manually:
```bash
ssh melvin@<JETSON_IP>
# Password: 123456
```

If that works, then run:
```bash
./deploy_manual.sh <JETSON_IP>
```


