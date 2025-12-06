# USB-to-CAN Motor Setup Guide

**Complete guide for connecting motors 12 and 14 via USB-to-CAN adapter with CH340 driver**

---

## 🎯 Overview

This setup enables control of Robstride motors 12 and 14 using a USB-to-CAN adapter (CH340 chipset) connected to the Jetson via USB.

**What you need:**
- USB-to-CAN adapter with CH340 chipset
- Jetson AGX Orin (or compatible)
- Motors 12 and 14 connected to CAN bus
- USB cable to connect adapter to Jetson

---

## 🚀 Quick Start

### On the Jetson (via USB connection):

```bash
# 1. Connect to Jetson via USB
#    Option A: SSH (if USB networking configured)
ssh melvin@192.168.55.1

#    Option B: USB serial terminal
./jetson_terminal.sh

# 2. Navigate to project directory
cd ~/melvin_motors  # or wherever you deployed

# 3. Run complete setup
sudo ./setup_usb_can_motors.sh
```

**That's it!** The script will:
- ✅ Load CH340 driver
- ✅ Detect USB serial device
- ✅ Setup CAN interface (slcand)
- ✅ Configure can0
- ✅ Compile motor test code
- ✅ Test motors 12 and 14

---

## 📋 Detailed Steps

### Step 1: Connect USB-to-CAN Adapter

1. Plug USB-to-CAN adapter into Jetson USB port
2. Connect CAN-H and CAN-L to motor bus
3. Verify adapter is detected:
   ```bash
   lsusb | grep -i ch340
   ls /dev/ttyUSB*
   ```

### Step 2: Run Setup Script

```bash
sudo ./setup_usb_can_motors.sh
```

The script will:
- Load CH340 driver (`modprobe ch341` or `ch340`)
- Find USB serial device (`/dev/ttyUSB0` or similar)
- Install can-utils if needed
- Setup slcand: `slcand -o -c -s6 /dev/ttyUSB0 can0`
- Configure CAN bitrate (125kbps)
- Bring can0 interface up
- Compile motor test code
- Run interactive test

### Step 3: Test Motors

The setup script will prompt to test motors. Or run manually:

```bash
sudo ./run_motors_12_14.sh
```

Or directly:
```bash
sudo ./test_motors_12_14
```

---

## 🔧 Manual Setup (if needed)

If the automated script doesn't work, here's manual setup:

### 1. Load CH340 Driver

```bash
sudo modprobe ch341
# or
sudo modprobe ch340
```

Verify:
```bash
lsmod | grep ch34
ls /dev/ttyUSB*
```

### 2. Install CAN Utilities

```bash
sudo apt-get update
sudo apt-get install -y can-utils
```

### 3. Setup CAN Interface

```bash
# Find USB device
SERIAL_DEVICE=$(ls /dev/ttyUSB* | head -1)

# Setup slcand (Serial Line CAN)
sudo slcand -o -c -s6 $SERIAL_DEVICE can0

# Wait for interface
sleep 2

# Configure and bring up
sudo ip link set can0 type can bitrate 125000
sudo ip link set can0 up

# Verify
ip link show can0
```

### 4. Test CAN

```bash
# Monitor CAN traffic
candump can0 &

# Send test frame
cansend can0 123#DEADBEEF
```

### 5. Compile Motor Test

```bash
gcc -O2 -Wall -o test_motors_12_14 test_motors_12_14.c -lm
```

### 6. Run Test

```bash
sudo ./test_motors_12_14
```

---

## 📦 Deployment from Mac/PC

### Option 1: Via SSH (USB Networking)

```bash
# Deploy everything
./deploy_usb_can_to_jetson.sh

# Then on Jetson
ssh melvin@192.168.55.1
cd ~/melvin_motors
sudo ./setup_usb_can_motors.sh
```

### Option 2: Via USB Serial

```bash
# Connect via USB serial
./jetson_terminal.sh

# On Jetson, manually copy files or use scp
```

### Option 3: Manual Copy

1. Copy files to USB drive or use scp
2. Extract on Jetson
3. Run setup script

---

## 🐛 Troubleshooting

### CH340 Driver Not Found

```bash
# Check if driver exists
modinfo ch341
modinfo ch340

# Install kernel modules
sudo apt-get install linux-modules-extra-$(uname -r)

# Try loading
sudo modprobe ch341
```

### No USB Device Found

```bash
# Check USB devices
lsusb

# Check dmesg for errors
dmesg | tail -20

# Verify permissions
ls -l /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
```

### CAN Interface Not Working

```bash
# Check if can0 exists
ip link show can0

# Check slcand process
ps aux | grep slcand

# Restart slcand
sudo pkill slcand
sudo slcand -o -c -s6 /dev/ttyUSB0 can0
sleep 2
sudo ip link set can0 up
```

### Motors Not Responding

1. **Check CAN bus:**
   ```bash
   candump can0
   # Should see traffic when motors are powered
   ```

2. **Verify motor IDs:**
   - Motor 12 = CAN ID 0x0C
   - Motor 14 = CAN ID 0x0E
   - Check in `test_motors_12_14.c`

3. **Check bitrate:**
   ```bash
   ip -details link show can0
   # Should match motor controller bitrate (usually 125kbps or 500kbps)
   ```

4. **Test with cansend:**
   ```bash
   # Enable motor 12
   cansend can0 0C#A100000000000000
   
   # Set position
   cansend can0 0C#A1[position_bytes][velocity_bytes][kp_bytes]
   ```

---

## 📝 Motor Test Code

The motor test code (`test_motors_12_14.c`) uses the Robstride protocol:

- **Motor 12**: CAN ID `0x0C` (12 decimal)
- **Motor 14**: CAN ID `0x0E` (14 decimal)
- **Protocol**: Robstride O2 MIT protocol
- **Commands**:
  - `0xA0`: Disable motor
  - `0xA1`: Enable motor / Position mode
  - Position: -12.5 to +12.5 radians
  - Velocity: 0 to 65 rad/s
  - KP: 0 to 500

---

## 🔄 Restarting CAN Interface

If you need to restart the CAN interface:

```bash
# Stop everything
sudo pkill slcand
sudo ip link set can0 down

# Restart
sudo slcand -o -c -s6 /dev/ttyUSB0 can0
sleep 2
sudo ip link set can0 up
```

Or use the quick script:
```bash
sudo ./run_motors_12_14.sh
# It will auto-setup if can0 is down
```

---

## ✅ Verification Checklist

- [ ] USB-to-CAN adapter plugged in
- [ ] CH340 driver loaded (`lsmod | grep ch34`)
- [ ] USB device detected (`ls /dev/ttyUSB*`)
- [ ] can-utils installed (`which slcand`)
- [ ] CAN interface up (`ip link show can0 | grep UP`)
- [ ] Motor test compiled (`./test_motors_12_14`)
- [ ] Motors powered and connected to CAN bus
- [ ] CAN traffic visible (`candump can0`)

---

## 🎯 Next Steps

Once motors 12 and 14 are working:

1. **Integrate with Melvin brain:**
   ```bash
   # Map motors to brain
   sudo ./tools/map_can_motors brain.m
   
   # Run motor runtime
   sudo ./melvin_motor_runtime brain.m
   ```

2. **Test other motors:**
   - Modify `test_motors_12_14.c` for other motor IDs
   - Or use `test_motor_exec` with different IDs

3. **Create motor control patterns:**
   - Brain will learn to control motors through experience
   - Teachable EXEC nodes will execute motor commands

---

## 📞 Quick Reference

**Setup:**
```bash
sudo ./setup_usb_can_motors.sh
```

**Test:**
```bash
sudo ./run_motors_12_14.sh
```

**Check status:**
```bash
ip link show can0
candump can0
```

**Restart:**
```bash
sudo pkill slcand
sudo slcand -o -c -s6 /dev/ttyUSB0 can0
sudo ip link set can0 up
```

---

## 🎉 Success!

When everything works, you should see:
- ✅ Motors 12 and 14 moving smoothly
- ✅ CAN traffic in `candump`
- ✅ No errors in motor test output
- ✅ Motors return to center position

**Motors are now ready for Melvin brain integration!** 🚀

