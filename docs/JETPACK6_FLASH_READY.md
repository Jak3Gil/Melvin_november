# JetPack 6 Flash - Ready to Go!

**All files downloaded - ready to flash Jetson AGX Orin**

---

## ✅ **What's Ready:**

Location: `/tmp/jetpack6_flash/`

- ✅ JetPack 6.0 BSP (648MB) - Downloaded
- ✅ Rootfs (1.7GB) - Downloaded & Extracted (5.4GB)
- ✅ Linux_for_Tegra directory - Ready
- ✅ flash.sh script - Ready

**Total size: ~6GB prepared**

---

## 🚀 **Flash Instructions (Run on Your Mac)**

### Step 1: Finish Preparation

```bash
cd /tmp/jetpack6_flash/Linux_for_Tegra

# Apply NVIDIA binaries
sudo ./apply_binaries.sh

# Should complete in ~1 minute
```

### Step 2: Put Jetson in Recovery Mode

**Physical steps on Jetson:**

1. **Power off** Jetson (unplug power)
2. **Locate buttons** on Jetson AGX Orin:
   - Recovery button (middle button)
   - Reset button (left button)  
   - Power button (right button)

3. **Enter recovery mode:**
   - Hold **Recovery** button
   - Press **Power** button (while holding Recovery)
   - Wait 2 seconds
   - Release **Recovery** button

4. **Connect USB-C** from your Mac to Jetson USB-C port

5. **Verify recovery mode:**
   ```bash
   # On your Mac
   system_profiler SPUSBDataType | grep NVIDIA
   # Should see: NVIDIA Corp. APX
   ```

### Step 3: Flash JetPack 6

```bash
# On your Mac
cd /tmp/jetpack6_flash/Linux_for_Tegra

# Flash command for Jetson AGX Orin
sudo ./flash.sh jetson-agx-orin-devkit mmcblk0p1
```

**This will:**
- Flash bootloader
- Flash kernel
- Flash rootfs
- Configure device tree (with CAN enabled!)
- **Takes ~20-30 minutes**

### Step 4: First Boot

After flash completes:
1. Jetson reboots automatically
2. Follow on-screen setup (need HDMI for this)
3. Create user account
4. Login

**Or skip GUI setup:**
```bash
# Default credentials after flash:
username: nvidia
password: nvidia

# SSH in:
ssh nvidia@169.254.123.100
```

---

## ✅ **After JetPack 6 Flash**

### Verify CAN is Enabled:

```bash
# SSH to Jetson
ssh nvidia@169.254.123.100

# Check CAN pins (should be CLAIMED now!)
cat /sys/kernel/debug/pinctrl/*/pinmux-pins | grep CAN0

# Expected:
# pin 138 (CAN0_DOUT): c310000.mttcan (CLAIMED) ✅
# pin 139 (CAN0_DIN):  c310000.mttcan (CLAIMED) ✅
```

### Test CAN Hardware:

```bash
# Configure CAN0
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# Monitor
candump can0 &

# Test send
cansend can0 123#DEADBEEF

# Should see frame in candump! ✅
```

### Test Motors:

```bash
# Send to motor 12
cansend can0 00C#A100000000000000

# Send to motor 14  
cansend can0 00E#A100000000000000

# Watch for responses and movement!
```

---

## 🎯 **Then Deploy Melvin Motor System:**

```bash
# On your Mac
cd /Users/jakegilbert/melvin_november/Melvin_november
./deploy_motors_to_jetson.sh

# On Jetson
cd ~/melvin_motors
sudo ./setup_jetson_motors.sh brain.m

# Test
sudo ./test_motor_exec brain.m all

# Run
sudo ./melvin_motor_runtime brain.m
```

**Motors will integrate with Melvin brain immediately!** 🎉

---

## ⚡ **Quick Start (TL;DR)**

```bash
# 1. On Mac - Prepare flash
cd /tmp/jetpack6_flash/Linux_for_Tegra
sudo ./apply_binaries.sh

# 2. Put Jetson in recovery mode (hold Recovery + press Power)

# 3. Flash
sudo ./flash.sh jetson-agx-orin-devkit mmcblk0p1

# 4. Wait 30 minutes

# 5. Test CAN
ssh nvidia@169.254.123.100
sudo ip link set can0 up
candump can0

# 6. Deploy motors
./deploy_motors_to_jetson.sh

# DONE! ✅
```

---

## 📍 **Current Status:**

**Files location:** `/tmp/jetpack6_flash/`
- All downloaded ✅
- BSP extracted ✅
- Rootfs extracted ✅  
- Ready to run `apply_binaries.sh` ⏳

**Ready when you are!** 

Just run:
```bash
cd /tmp/jetpack6_flash/Linux_for_Tegra
sudo ./apply_binaries.sh
# Then follow recovery mode steps above
```

🚀 **Want me to create a single automated flash script?**


