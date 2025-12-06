# Flash JetPack 6 to Jetson AGX Orin

**Complete guide to flashing JetPack 6.0 for full CAN/motor support**

---

## 🎯 What JetPack 6 Gives You

**After flashing:**
- ✅ CAN0 and CAN1 fully enabled
- ✅ All GPIO pins accessible
- ✅ Full desktop environment
- ✅ Display working out of box
- ✅ All development tools
- ✅ Latest CUDA and libraries

**Basically: Everything works!**

---

## 📋 Requirements

### On Your Mac:

1. **Ubuntu VM or Ubuntu machine** (SDK Manager needs Ubuntu)
   - Or use command-line flash method (Mac compatible)

2. **USB-C cable** (Mac to Jetson)
   - For recovery mode connection

3. **Internet connection** (to download JetPack)

### On Jetson:

1. **Power supply** connected
2. **Recovery mode** access (button or jumper)

---

## 🚀 Method 1: SDK Manager (Easiest)

### Step 1: Download SDK Manager

**On Ubuntu machine:**
```bash
# Download from NVIDIA
wget https://developer.nvidia.com/downloads/sdkmanager

# Or get latest version
# Visit: https://developer.nvidia.com/sdk-manager
```

### Step 2: Install SDK Manager

```bash
sudo apt install ./sdkmanager_[version].deb
```

### Step 3: Put Jetson in Recovery Mode

**For Jetson AGX Orin:**
1. Power off Jetson
2. Hold **Recovery button** (or connect FC_REC to GND)
3. Press **Power button**
4. Connect USB-C from Mac/Ubuntu to Jetson
5. Release recovery button after 2 seconds

**Verify recovery mode:**
```bash
lsusb | grep NVIDIA
# Should see: NVIDIA Corp. APX
```

### Step 4: Flash with SDK Manager

1. Run: `sdkmanager`
2. Login with NVIDIA account
3. Select:
   - Product: Jetson AGX Orin
   - Version: **JetPack 6.0**
   - Target components: **Jetson OS** (minimum)
4. Click "Continue" and "Flash"
5. Wait ~30 minutes

### Step 5: First Boot

After flash:
1. Jetson reboots
2. Follow on-screen setup
3. Create user account
4. **CAN will be enabled!**

---

## ⚡ Method 2: Command Line Flash (From Mac!)

**This works from macOS!**

### Step 1: Download BSP

```bash
# On your Mac
cd ~/Downloads

# Download JetPack 6.0 BSP
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/jetson_linux_r36.3.0_aarch64.tbz2

# Download sample rootfs
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/tegra_linux_sample-root-filesystem_r36.3.0_aarch64.tbz2
```

### Step 2: Prepare Flash Files

```bash
# Extract BSP
tar xf jetson_linux_r36.3.0_aarch64.tbz2
cd Linux_for_Tegra

# Extract rootfs
sudo tar xpf ../tegra_linux_sample-root-filesystem_r36.3.0_aarch64.tbz2 -C rootfs/

# Apply binaries
sudo ./apply_binaries.sh
```

### Step 3: Put Jetson in Recovery

Same as above - connect USB-C while in recovery mode.

### Step 4: Flash

```bash
# For Jetson AGX Orin DevKit:
sudo ./flash.sh jetson-agx-orin-devkit mmcblk0p1

# Wait ~20 minutes
```

---

## 🔧 Method 3: Simplified Flash (Recommended for You!)

Since you're on Mac and have minimal time:

### **Use NVIDIA's Flash Tool:**

```bash
# On your Mac

# 1. Download latest JetPack 6 image
# Visit: https://developer.nvidia.com/embedded/jetpack

# 2. Use balenaEtcher or dd to flash SD card (if using SD)
# OR use NVIDIA L4T flash script

# 3. Put Jetson in recovery mode

# 4. Run flash command
sudo ./flash.sh jetson-agx-orin-devkit mmcblk0p1
```

---

## ✅ After Flashing JetPack 6

### Verify CAN is Enabled:

```bash
# SSH to Jetson
ssh melvin@169.254.123.100

# Check CAN pins
cat /sys/kernel/debug/pinctrl/*/pinmux-pins | grep CAN0

# Should show:
# pin 138 (CAN0_DOUT): c310000.mttcan (CLAIMED) ✅
# pin 139 (CAN0_DIN):  c310000.mttcan (CLAIMED) ✅
```

### Test Motors Immediately:

```bash
# Configure CAN
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# Monitor
candump can0 &

# Test motor 12
cansend can0 00C#A100000000000000

# Should see CAN traffic and motor response!
```

### Deploy Melvin Motor Integration:

```bash
# On your Mac
cd /Users/jakegilbert/melvin_november/Melvin_november
./deploy_motors_to_jetson.sh

# On Jetson (after deploy)
cd ~/melvin_motors
sudo ./setup_jetson_motors.sh brain.m

# Motors will work immediately!
```

---

## 📦 Quick Start Commands

### On Mac (Download JetPack):

```bash
# Visit NVIDIA site
open https://developer.nvidia.com/embedded/jetpack

# Or direct download (requires NVIDIA login)
# JetPack 6.0 for Jetson AGX Orin
```

### Put Jetson in Recovery:

```bash
# 1. Power off Jetson
# 2. Hold Recovery button
# 3. Press Power
# 4. Connect USB-C to Mac
# 5. Release Recovery after 2 sec
# 6. Verify: lsusb (should see NVIDIA APX)
```

### Flash:

```bash
cd Linux_for_Tegra
sudo ./flash.sh jetson-agx-orin-devkit mmcblk0p1
# Wait 20-30 minutes
```

---

## 🎉 Result

After JetPack 6 flash:
- ✅ CAN0 enabled and working
- ✅ Full desktop environment
- ✅ All development tools
- ✅ Display works perfectly
- ✅ **Motors will work immediately with all our code!**

---

**Ready to flash? I can guide you through each step!** 🚀

Would you like me to:
1. Create detailed flash commands for your Mac?
2. Prepare post-flash motor test scripts?
3. Both?

