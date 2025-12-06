# Jetson AGX Orin Native CAN Setup

**Using Jetson's built-in MTTCAN controller with external CAN transceiver**

---

## 🎯 Why Native CAN?

**Problems with Robstride USB-CAN Adapter:**
- ❌ Uses proprietary L91 protocol
- ❌ Needs Windows software (Motor Studio)
- ❌ Not standard Linux SocketCAN
- ❌ Difficult to integrate

**Benefits of Native CAN:**
- ✅ Standard Linux SocketCAN
- ✅ No proprietary protocols
- ✅ Works with all our code
- ✅ Better performance
- ✅ Direct Jetson integration

---

## 🔧 Hardware Requirements

### 1. CAN Transceiver Board

**Recommended:** SN65HVD230 CAN Transceiver Module

**Where to buy:**
- Amazon: ~$3-5
- AliExpress: ~$1-2
- Search: "SN65HVD230 CAN module"

**Specifications:**
- 3.3V logic (perfect for Jetson)
- 1 Mbps max bitrate
- Built-in protection
- Small form factor

### 2. Wiring Materials

- Jumper wires (female-to-female for Jetson header)
- Twisted pair cable for CAN bus (or CAT5 cable)
- 2× 120Ω resistors (for bus termination)
- Heat shrink or electrical tape

---

## 📍 Pin Connections

### Jetson AGX Orin J30 Header (40-pin)

```
Looking at J30 header with pins facing you:

Pin 1  [3.3V]        Pin 2  [5V]
Pin 3  [I2C]         Pin 4  [5V]
...
Pin 29 [CAN0_DIN]    Pin 30 [GND]  ← CAN0 RX
Pin 31 [CAN0_DOUT]   Pin 32 [GPIO] ← CAN0 TX
Pin 33 [CAN1_DOUT]   Pin 34 [GND]  ← CAN1 TX  
...
Pin 37 [CAN1_DIN]    Pin 38 [GPIO] ← CAN1 RX
Pin 39 [GND]         Pin 40 [GPIO]
```

### Wiring Table

| Jetson Pin | Signal | Wire To | SN65HVD230 Pin |
|------------|--------|---------|----------------|
| **Pin 1** | 3.3V | → | **VCC** (3.3V power) |
| **Pin 6** | GND | → | **GND** (ground) |
| **Pin 29** | CAN0_DIN | → | **CRX** (RX output to Jetson) |
| **Pin 31** | CAN0_DOUT | → | **CTX** (TX input from Jetson) |

| SN65HVD230 Pin | Wire To | Motor CAN Bus |
|----------------|---------|---------------|
| **CANH** | → | **CAN-H** (to all 14 motors) |
| **CANL** | → | **CAN-L** (to all 14 motors) |

---

## 🔌 Step-by-Step Wiring

### Step 1: Connect Power to Transceiver

1. Jetson Pin 1 (3.3V) → SN65HVD230 VCC
2. Jetson Pin 6 (GND) → SN65HVD230 GND

### Step 2: Connect CAN Signals

3. Jetson Pin 31 (CAN0_DOUT/TX) → SN65HVD230 CTX
4. Jetson Pin 29 (CAN0_DIN/RX) → SN65HVD230 CRX

### Step 3: Connect to Motor Bus

5. SN65HVD230 CANH → Motor Bus CAN-H
6. SN65HVD230 CANL → Motor Bus CAN-L

### Step 4: Add Termination

7. 120Ω resistor between CAN-H and CAN-L at START of motor bus
8. 120Ω resistor between CAN-H and CAN-L at END of motor bus

**Verify:** Measure 60Ω between CAN-H and CAN-L (with motors powered off)

---

## ⚙️ Software Setup

### Enable CAN0 Interface

```bash
# Configure CAN0 for 500kbps (Robstride motors)
sudo ip link set can0 type can bitrate 500000 restart-ms 100
sudo ip link set can0 up

# Verify
ip -details link show can0
```

Expected output:
```
can state ERROR-ACTIVE (berr-counter tx 0 rx 0) restart-ms 100 
bitrate 500000 sample-point 0.870
```

### Test CAN Communication

```bash
# Monitor CAN traffic
candump can0 &

# Send test frame
cansend can0 00C#FFFFFFFFFFFFFFFC

# Should see frame on candump
```

---

## 🧪 Testing Motors

### Test 1: Scan for Motors

```bash
cd /home/melvin/melvin_motors
sudo ./tools/map_can_motors brain.m
```

Should detect all 14 motors!

### Test 2: Test Individual Motor

```bash
sudo ./test_motor_exec brain.m 12
```

Motor 12 should move!

### Test 3: Full System

```bash
sudo ./melvin_motor_runtime brain.m
```

Motors now controlled by Melvin brain! 🎉

---

## 🔒 Safety Notes

**Before Powering On:**

1. ✅ Double-check all wiring
2. ✅ Verify 3.3V to VCC (NOT 5V!)
3. ✅ Confirm GND connections
4. ✅ Check CAN-H/CAN-L polarity
5. ✅ Verify 120Ω termination

**Never:**
- ❌ Connect 5V to SN65HVD230 VCC (will damage it)
- ❌ Hot-plug CAN transceiver (power off first)
- ❌ Reverse CAN-H and CAN-L

---

## 🐛 Troubleshooting

### CAN Interface Not Found

```bash
# Check if CAN is enabled in device tree
ls /sys/class/net/can*

# If not found, may need to enable in device tree
# Use jetson-io.py to configure CAN
sudo /opt/nvidia/jetson-io/jetson-io.py
```

### No Motor Responses

1. Check transceiver power LED
2. Verify wiring with multimeter
3. Check termination (should be 60Ω)
4. Try lower bitrate: `bitrate 250000`

### CAN Bus Errors

```bash
# Check error counters
ip -statistics link show can0

# Reset CAN interface
sudo ip link set can0 down
sudo ip link set can0 up
```

---

## 📊 Expected Results

### With Proper Setup:

```bash
$ sudo ./tools/map_can_motors brain.m

Scanning CAN bus...
✅ Motor 0 detected (CAN ID 0x01)
✅ Motor 1 detected (CAN ID 0x02)
...
✅ Motor 13 detected (CAN ID 0x0E)

Found 14 motors!
Mapping to brain ports...
✅ Complete!
```

---

## 🚀 Integration with Melvin

Once CAN is working:

1. **Map motors:**
   ```bash
   sudo ./tools/map_can_motors brain.m
   ```

2. **Test motor control:**
   ```bash
   sudo ./test_motor_exec brain.m 12
   ```

3. **Run motor runtime:**
   ```bash
   sudo ./melvin_motor_runtime brain.m
   ```

4. **Integrate with sensors:**
   ```bash
   python3 multimodal_brain_integration.py brain.m
   ```

**Brain learns motor control automatically!** ✨

---

## 💡 Shopping List

To get this working:

1. **SN65HVD230 CAN Transceiver Board** - $3-5
   - Or any 3.3V CAN transceiver
   
2. **Jumper Wires** - $5
   - Female-to-female for Jetson header
   
3. **120Ω Resistors** (if you don't have) - $1
   - 1/4W through-hole resistors

**Total: ~$10** 

**Time to setup: 15 minutes**

---

## ✅ Advantages

**vs Robstride USB Adapter:**

| Feature | Robstride USB | Native CAN |
|---------|---------------|------------|
| Linux Support | ❌ Proprietary | ✅ Standard |
| SocketCAN | ❌ No | ✅ Yes |
| Latency | Higher | Lower |
| Integration | Difficult | Easy |
| Cost | Expensive | ~$10 |
| Reliability | Unknown | Proven |

**Native CAN is clearly the better choice!**

---

## 🎯 Next Steps

1. **Order SN65HVD230 board** (or use what you have)
2. **Wire according to table above**
3. **Enable can0 on Jetson**
4. **Run our motor code** - it will just work!
5. **Integrate with Melvin brain**

**All the code is ready - just need the hardware connection!** 🚀

---

## 📞 Quick Reference

**Enable CAN:**
```bash
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

**Test:**
```bash
candump can0 &
cansend can0 001#A100000000000000
```

**Use with Melvin:**
```bash
sudo ./melvin_motor_runtime brain.m
```

**Done!** 🎉


