# Motor Integration - Quick Start

**Get 14 CAN motors working with Melvin in 5 minutes**

---

## 🚀 Setup (One Command)

On your **development machine**:

```bash
# Deploy to Jetson
./deploy_motors_to_jetson.sh
```

On the **Jetson** (or SSH in):

```bash
# Setup motors
cd ~/melvin_motors
sudo ./setup_jetson_motors.sh brain.m
```

That's it! 🎉

---

## 🧪 Test

```bash
# Test all motors
sudo ./test_motor_exec brain.m all

# Test specific motor
sudo ./test_motor_exec brain.m 0 seq
```

---

## ▶️ Run

```bash
# Start motor runtime
sudo ./melvin_motor_runtime brain.m
```

The brain will now:
- ✅ Monitor sensory inputs
- ✅ Discover patterns
- ✅ Learn to control motors
- ✅ Execute motor commands automatically

---

## 📊 Monitor

```bash
# Watch CAN traffic
candump can0

# Monitor motor runtime
tail -f melvin_motor.log

# Check brain state
./tools/inspect_graph brain.m | grep MOTOR
```

---

## 🛑 Stop

Press `Ctrl+C` in the motor runtime terminal.

Motors will safely stop and brain state will be saved.

---

## 🔧 Configuration

### Change CAN Bitrate

```bash
# Before setup
export CAN_BITRATE=250000
sudo ./setup_jetson_motors.sh brain.m
```

### Change Motor IDs

Edit `motor_config.txt`:

```yaml
motor_0:
  can_id: 0x10  # Change this
  ...
```

Then restart motor runtime.

---

## 🐛 Troubleshooting

### No CAN Interface

```bash
# Check USB adapter
lsusb

# Setup manually
sudo ip link set can0 type can bitrate 125000
sudo ip link set can0 up
```

### No Motors Found

1. Check power to motors
2. Verify CAN wiring
3. Test manually:
   ```bash
   cansend can0 001#10
   candump can0
   ```

### Permission Denied

```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER
# Logout and login again
```

---

## 📖 Full Documentation

See `MOTOR_INTEGRATION.md` for:
- Detailed architecture
- ARM64 code explanation
- Learning algorithm details
- Advanced features
- Safety considerations

---

## 🎯 What's Happening?

### The Magic ✨

1. **You don't program motor control**
2. **Brain discovers patterns** from sensors
3. **Brain learns routing** to motor EXEC nodes
4. **EXEC nodes execute** ARM64 motor code
5. **Motors move** based on learned patterns

**Example**:
```
Person waves
  → Camera: "HAND_GESTURE"
  → Brain discovers pattern
  → Routes to motor_0_exec
  → ARM64 code executes
  → Robot waves back!
  
All learned, not programmed! 🤖✨
```

---

## 💡 Next Steps

1. **Add sensors**: Camera, microphone
2. **Feed data**: Let brain see and hear
3. **Provide feedback**: Help brain learn
4. **Watch it learn**: Brain discovers motor control!

---

**Questions?** See full docs: `MOTOR_INTEGRATION.md`

**Ready to deploy?** Run: `./deploy_motors_to_jetson.sh`

🚀 Let's teach your robot to move!

