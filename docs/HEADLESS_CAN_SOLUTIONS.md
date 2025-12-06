# Headless Jetson CAN Solutions

**3 ways to get CAN working without display**

---

## 🎯 **Option 1: Command-Line Device Tree Flash (FASTEST)**

Since you can't use GUI, manually flash the device tree:

### **On the Jetson:**

```bash
# 1. Download CAN-enabled device tree from NVIDIA
cd /tmp
wget https://developer.nvidia.com/downloads/embedded/l4t/r35_release_v4.1/sources/public_sources.tbz2

# 2. Or use existing kernel and add CAN overlay
sudo su
cd /boot

# 3. Backup current DTB
cp kernel_tegra234-p3701-0005-p3737-0000-user-custom.dtb kernel_original.dtb.backup

# 4. The nuclear option - enable ALL pins in pinmux
# This requires kernel rebuild or NVIDIA flash tools
```

**This is complex for headless...**

---

## ✅ **Option 2: MCP2515 SPI-CAN Module (RECOMMENDED!)**

**Best solution for headless:** Use SPI instead of GPIO CAN pins!

### **Why This Works:**

- SPI pins are usually enabled by default
- No device tree changes needed
- MCP2515 driver built into kernel
- Creates standard `can0` interface
- $3-5 on Amazon

### **Hardware:**

**Buy:** MCP2515 CAN Bus Module
- Search Amazon: "MCP2515 TJA1050 CAN Module"
- ~$3-5 with free shipping

### **Wiring to Jetson J30:**

```
MCP2515 Pin    →    Jetson J30 Pin
───────────         ──────────────
VCC            →    Pin 1  (3.3V or 5V - check module)
GND            →    Pin 6  (GND)
SI (MOSI)      →    Pin 19 (SPI1_MOSI)
SO (MISO)      →    Pin 21 (SPI1_MISO)
SCK            →    Pin 23 (SPI1_CLK)
CS             →    Pin 24 (SPI1_CS0)
INT            →    Pin 22 (GPIO25) [optional]

CANH           →    Motor CAN-H bus
CANL           →    Motor CAN-L bus
```

### **Software Setup (Headless SSH!):**

```bash
# Load MCP2515 driver
sudo modprobe mcp251x

# Configure SPI CAN
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# Test
candump can0 &
cansend can0 00C#A100000000000000
```

**Should work immediately!**

---

## 💡 **Option 3: Standard USB-CAN Adapter**

Replace Robstride adapter with Linux-compatible USB-CAN:

### **Compatible Adapters:**

1. **CANable** ($25)
   - Open source
   - gs_usb driver (built into Linux)
   - Plug and play

2. **PEAK PCAN-USB** ($50)
   - Industrial grade
   - Perfect Linux support
   - Very reliable

3. **Kvaser Leaf Light** ($200)
   - Professional grade
   - Best Linux support

### **Setup:**

```bash
# Plug in adapter
# Automatically creates can0

sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# Works immediately!
```

---

## 🚀 **My Recommendation**

**For your headless Jetson: Get MCP2515 module**

**Advantages:**
- ✅ Works via SPI (no GPIO mux needed)
- ✅ Only $3-5
- ✅ Can setup via SSH
- ✅ All our code works unchanged
- ✅ 2-day shipping on Amazon

**Disadvantages:**
- ⏱️ Have to wait for shipping
- 🔧 Need to wire 6 pins

---

## 📦 **What To Order**

**Amazon:** Search "MCP2515 CAN module"

**Get one with:**
- TJA1050 or SN65HVD230 transceiver
- 3.3V or 5V compatible
- SPI interface
- Screw terminals for CAN-H/CAN-L

**Price:** $3-8
**Shipping:** 1-2 days with Prime

---

## 🔧 **Setup Script Ready**

I'll create complete setup script for MCP2515 that works via SSH:
- Auto-detects SPI
- Loads driver  
- Configures can0
- Tests motors
- Integrates with Melvin

**Want me to create the MCP2515 integration now?**

Then when it arrives, plug it in and run the script - motors will work! 🚀

---

## ⚡ **Alternative: Physical Access**

If you can get to the Jetson physically:

1. Connect HDMI monitor
2. Connect USB keyboard  
3. Reboot Jetson
4. Run: `sudo /opt/nvidia/jetson-io/jetson-io.py`
5. Enable CAN0
6. Reboot
7. Done!

**Takes 5 minutes with physical access.**

---

**Which option do you prefer?**
- A) Order MCP2515 module ($5, 2 days)
- B) Get physical access to Jetson (5 minutes)
- C) Order standard USB-CAN adapter ($25-50)

