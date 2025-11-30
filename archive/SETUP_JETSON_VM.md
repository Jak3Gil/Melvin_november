# Setup Linux VM to Flash JetPack on Jetson

## Overview
This guide will help you set up a Linux VM on your Mac to use NVIDIA SDK Manager to flash JetPack 6.x on your Jetson Orin AGX.

## Prerequisites

1. **Virtualization Software** (choose one):
   - **VirtualBox** (free): https://www.virtualbox.org/
   - **VMware Fusion** (paid): https://www.vmware.com/products/fusion.html
   - **UTM** (free, Apple Silicon): https://mac.getutm.app/

2. **Ubuntu 22.04 Desktop ISO**:
   - Download: https://releases.ubuntu.com/22.04/ubuntu-22.04.3-desktop-amd64.iso

3. **Jetson Requirements**:
   - Jetson in Recovery Mode (RCM)
   - USB cable connected to Mac
   - At least 50GB free disk space for VM

## Step 1: Create Ubuntu 22.04 VM

### Using VirtualBox:

```bash
# Install VirtualBox (if not installed)
brew install --cask virtualbox

# Create new VM:
# 1. Open VirtualBox
# 2. Click "New"
# 3. Name: "Ubuntu-Jetson-Flash"
# 4. Type: Linux
# 5. Version: Ubuntu (64-bit)
# 6. Memory: 4096 MB (4GB minimum)
# 7. Hard disk: 50GB (dynamically allocated)
# 8. Click "Create"

# Configure VM:
# 1. Select VM → Settings
# 2. System → Processor: 2 CPUs
# 3. Display → Video Memory: 128MB
# 4. USB → Enable USB Controller (USB 3.0)
# 5. Storage → Add Ubuntu 22.04 ISO as optical drive
# 6. Click "OK"
```

### Using UTM (Apple Silicon Macs):

```bash
# Install UTM
brew install --cask utm

# Create VM:
# 1. Open UTM
# 2. Click "+" → "Virtualize"
# 3. Select "Linux"
# 4. Choose Ubuntu 22.04 ISO
# 5. Memory: 4096 MB
# 6. Storage: 50GB
# 7. Enable USB sharing
```

## Step 2: Install Ubuntu 22.04 in VM

1. Start the VM
2. Boot from Ubuntu ISO
3. Follow installation wizard:
   - Language: English
   - Installation type: Normal installation
   - Create user account
   - Wait for installation to complete
4. Reboot VM

## Step 3: Install SDK Manager in VM

Once Ubuntu is installed and running in the VM:

```bash
# Open terminal in VM (Ctrl+Alt+T)

# Update system
sudo apt update
sudo apt upgrade -y

# Install SDK Manager
# Option 1: From NVIDIA repository (recommended)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install sdkmanager

# Option 2: Download .deb directly
# wget https://developer.download.nvidia.com/sdkmanager/redirects/sdkmanager-deb.html
# sudo apt install ./sdkmanager_*.deb

# Verify installation
sdkmanager --version
```

## Step 4: Connect Jetson to VM

### Put Jetson in Recovery Mode:

1. **Power off Jetson** (if running)
2. **Press and hold** the RECOVERY button on Jetson
3. **While holding**, press and release the RESET button
4. **Keep holding RECOVERY** for 2 seconds, then release
5. Jetson is now in Recovery Mode (RCM)

### Pass USB through to VM:

#### VirtualBox:
1. With VM running, click **Devices → USB**
2. Select **NVIDIA Corp. APX** (Jetson in recovery mode)
3. USB device is now passed to VM

#### UTM:
1. With VM running, click **USB** in toolbar
2. Select Jetson USB device
3. Device is passed to VM

### Verify Jetson is detected in VM:

```bash
# In VM terminal
lsusb | grep -i nvidia

# Should show:
# Bus XXX Device XXX: ID 0955:7f21 NVIDIA Corp. APX
```

## Step 5: Flash JetPack 6.x

### Using SDK Manager GUI:

```bash
# Launch SDK Manager
sdkmanager

# In SDK Manager:
# 1. Select "Jetson" tab
# 2. Choose "Jetson AGX Orin"
# 3. Select JetPack 6.2 (or latest 6.x)
# 4. Check "Flash OS" and "Install SDK Components"
# 5. Click "CONTINUE"
# 6. Follow wizard to flash Jetson
```

### Using SDK Manager CLI:

```bash
# Flash JetPack 6.2 via command line
sdkmanager --cli install \
  --logintype devzone \
  --product Jetson \
  --target JETSON_AGX_ORIN \
  --version 6.2 \
  --targetos Linux \
  --flash all \
  --license accept

# Follow prompts to:
# 1. Login to NVIDIA Developer account
# 2. Select installation options
# 3. Flash Jetson
```

## Step 6: Configure Display After Flash

After flashing JetPack 6.x, you can configure display during first boot:

1. Boot Jetson
2. During setup, configure display resolution
3. Or edit `/boot/extlinux/extlinux.conf` after first boot:

```bash
# SSH to Jetson
ssh melvin@169.254.123.100

# Edit boot config
sudo nano /boot/extlinux/extlinux.conf

# Add to APPEND line:
video=HDMI-A-1:1024x768@60

# Save and reboot
sudo reboot
```

## Troubleshooting

### Jetson not detected in VM:
- Ensure Jetson is in Recovery Mode
- Check USB cable (use high-quality cable)
- Try different USB port on Mac
- In VirtualBox: Devices → USB → Enable USB 3.0 Controller

### SDK Manager login issues:
- Create NVIDIA Developer account: https://developer.nvidia.com/
- Use `--logintype devzone` for CLI

### Flash fails:
- Ensure Jetson is in Recovery Mode
- Check USB connection
- Try different USB port
- Restart VM and try again

### Display still shows "resolution not supported":
- After flashing JetPack 6.x, try different resolutions
- Check monitor specifications
- Consider HDMI scaler as hardware solution

## Next Steps

After successfully flashing JetPack 6.x:
1. Boot Jetson
2. Configure display resolution
3. Reinstall Melvin system
4. Test display output

## References

- NVIDIA SDK Manager: https://developer.nvidia.com/sdk-manager
- Jetson Linux Developer Guide: https://docs.nvidia.com/jetson/
- JetPack Release Notes: https://docs.nvidia.com/jetson/jetpack/

