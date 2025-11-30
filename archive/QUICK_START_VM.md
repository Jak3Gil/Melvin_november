# Quick Start: Linux VM for Jetson Flashing

## Step 1: Install VirtualBox

**Manual Installation (requires password):**

1. Open Terminal
2. Run: `brew install --cask virtualbox`
3. Enter your Mac password when prompted
4. Wait for installation to complete

**Or download manually:**
- Go to: https://www.virtualbox.org/wiki/Downloads
- Download: VirtualBox for macOS (ARM64 if Apple Silicon, or Intel)
- Install the .dmg file

## Step 2: Download Ubuntu 22.04

**Option A: Browser Download (Recommended)**
1. Open browser
2. Go to: https://releases.ubuntu.com/22.04/
3. Click: "ubuntu-22.04.3-desktop-amd64.iso" (4.6 GB)
4. Save to: `~/Downloads/`

**Option B: Terminal Download**
```bash
cd ~/Downloads
curl -L -o ubuntu-22.04.3-desktop-amd64.iso \
  https://releases.ubuntu.com/22.04/ubuntu-22.04.3-desktop-amd64.iso
```

## Step 3: Create VM

**Option A: Use Automated Script**
```bash
cd /Users/jakegilbert/melvin_november/Melvin_november
./create_vm.sh
```

**Option B: Manual Creation in VirtualBox**
1. Open VirtualBox
2. Click "New"
3. Name: `Ubuntu-Jetson-Flash`
4. Type: Linux
5. Version: Ubuntu (64-bit)
6. Memory: 4096 MB
7. Hard disk: 50 GB (dynamically allocated)
8. Click "Create"
9. Settings → System → Processor: 2 CPUs
10. Settings → USB → Enable USB Controller (USB 3.0)
11. Settings → Storage → Add Ubuntu ISO as optical drive

## Step 4: Install Ubuntu in VM

1. Start VM
2. Boot from Ubuntu ISO
3. Follow installation wizard
4. Create user account
5. Wait for installation (~20 minutes)
6. Reboot VM

## Step 5: Install SDK Manager in VM

Once Ubuntu is running in VM:

```bash
# Open terminal (Ctrl+Alt+T)

# Update system
sudo apt update
sudo apt upgrade -y

# Install SDK Manager
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install sdkmanager

# Verify
sdkmanager --version
```

## Step 6: Put Jetson in Recovery Mode

1. **Power off Jetson** completely
2. **Press and hold** the RECOVERY button (small button on Jetson)
3. **While holding RECOVERY**, press and release the RESET button
4. **Keep holding RECOVERY** for 2 seconds, then release
5. Jetson is now in Recovery Mode

## Step 7: Connect Jetson to VM

1. Connect Jetson to Mac via USB cable
2. In VirtualBox (with VM running):
   - Click **Devices → USB**
   - Select **NVIDIA Corp. APX** (Jetson in recovery)
3. Verify in VM:
   ```bash
   lsusb | grep -i nvidia
   # Should show: Bus XXX Device XXX: ID 0955:7f21 NVIDIA Corp. APX
   ```

## Step 8: Flash JetPack 6.x

**Using SDK Manager GUI:**
```bash
sdkmanager
```

1. Select "Jetson" tab
2. Choose "Jetson AGX Orin"
3. Select JetPack 6.2 (or latest 6.x)
4. Check "Flash OS" and "Install SDK Components"
5. Click "CONTINUE"
6. Login to NVIDIA Developer account
7. Follow wizard to flash Jetson

**Using SDK Manager CLI:**
```bash
sdkmanager --cli install \
  --logintype devzone \
  --product Jetson \
  --target JETSON_AGX_ORIN \
  --version 6.2 \
  --targetos Linux \
  --flash all \
  --license accept
```

## Step 9: Configure Display After Flash

After flashing, configure display resolution:

```bash
# SSH to Jetson (new IP after flash)
ssh melvin@<new_ip>

# Edit boot config
sudo nano /boot/extlinux/extlinux.conf

# Add to APPEND line:
video=HDMI-A-1:1024x768@60

# Save and reboot
sudo reboot
```

## Troubleshooting

**VirtualBox installation fails:**
- Run manually: `brew install --cask virtualbox`
- Enter password when prompted
- Or download from virtualbox.org

**Jetson not detected:**
- Ensure Jetson is in Recovery Mode
- Try different USB port
- In VirtualBox: Devices → USB → Enable USB 3.0 Controller

**SDK Manager login:**
- Create account: https://developer.nvidia.com/
- Use `--logintype devzone` for CLI

## Next Steps

After flashing JetPack 6.x:
1. Boot Jetson
2. Configure display
3. Reinstall Melvin
4. Test display output

See `SETUP_JETSON_VM.md` for detailed instructions.

