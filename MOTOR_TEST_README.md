# Motor 12 & 14 Test Guide

**Safe, slow testing of motors 12 and 14 to discover what they control**

---

## 🎯 Goal

Slowly move motors 12 and 14 to observe what physical components they control, documenting findings for the full motor mapping.

---

## 🚀 Quick Start

### On Your Mac (Deploy)

```bash
./deploy_motor_test.sh
```

### On Jetson (Run Test)

```bash
ssh melvin@192.168.55.1
cd /tmp
sudo ./test_motors_interactive.sh
```

---

## 🔒 Safety Features

The test program includes:

1. **Very Slow Movement**: 500ms between steps (0% → 5% → 10% → ...)
2. **Small Steps**: 5% increments
3. **Limited Range**: Only moves 0% → 50% → 0%
4. **Manual Control**: Press Enter to start each motor
5. **Emergency Stop**: Ctrl+C stops immediately and disables motors
6. **Safe Shutdown**: Always stops motors before exit

---

## 📋 What the Test Does

### Phase 1: Motor 12 Test

```
1. Slow movement from 0% → 50%
   • 5% steps
   • 500ms delay between steps
   • Watch what moves!

2. Hold at 50% for 2 seconds
   • Observe the position

3. Slow return to 0%
   • Same 5% steps
   • Returns to starting position

4. Stop and disable motor
```

### Phase 2: Motor 14 Test

Same process as Motor 12.

---

## 👁️ What to Observe

While each motor moves, watch for:

- **Which component moves**: Wheel? Arm? Gripper? Head?
- **Direction**: Clockwise? Counter-clockwise? Forward? Back?
- **Range of motion**: Full rotation? Limited arc? Linear?
- **Sound**: Normal operation? Grinding? Smooth?
- **Resistance**: Does it struggle or move freely?

---

## 📝 Recording Results

The script will prompt you to describe what each motor does:

```
What did Motor 12 control? (e.g., 'left wheel', 'gripper'): 
```

Your answers are saved to `motor_observations.txt`.

---

## 🔧 Technical Details

### CAN Protocol

Motors use standard CAN frame:
```
CAN ID: 0x01 + motor_id
  Motor 12 = 0x0D
  Motor 14 = 0x0F

Data Frame:
  Byte 0: Command (0x01 = position)
  Bytes 1-4: Position (float, 0.0-1.0)
  Bytes 5-7: Speed/safety limits (0x00 = slow)
```

### Movement Pattern

```
Position:  0% ─→ 5% ─→ 10% ─→ ... ─→ 50% ─→ [hold] ─→ 45% ─→ ... ─→ 0%
Time:      0s    0.5s   1.0s        5.0s    7.0s        7.5s        12s
```

---

## 🐛 Troubleshooting

### CAN Interface Not Ready

```bash
# Check status
ip link show can0

# Configure and bring up
sudo ip link set can0 type can bitrate 125000
sudo ip link set can0 up

# Verify
ip -details link show can0
```

### Motor Doesn't Move

1. **Check power**: Is motor powered?
2. **Check wiring**: CAN-H and CAN-L connected?
3. **Check termination**: 120Ω resistors at both ends?
4. **Monitor CAN**: 
   ```bash
   candump can0  # Should see frames being sent
   ```
5. **Check motor ID**: Verify actual CAN ID matches expectation

### Motor Moves Unexpectedly

**IMMEDIATELY**: Press Ctrl+C or hit E-stop!

Then:
1. Check for loose connections
2. Verify CAN ID is correct
3. Check for conflicting controllers on bus
4. Test motor manually with `cansend`

---

## 📊 Example Output

```
╔═══════════════════════════════════════════╗
║   Motor 12 & 14 Test - Slow Movement     ║
║   Press Ctrl+C at any time to stop       ║
╚═══════════════════════════════════════════╝

📡 Initializing CAN interface: can0
✅ CAN interface ready

⚠️  SAFETY CHECK:
   • Is the robot in a safe position?
   • Are there obstacles in the way?
   • Is someone ready to hit E-stop?

Press Enter to start testing Motor 12...

═══════════════════════════════════════════
Testing Motor 12 (Motor 12)
═══════════════════════════════════════════

🔹 Phase 1: Slow movement from 0% → 50%
   Watch what moves on the robot!

   Position: 0% ✓ Sent
   Position: 5% ✓ Sent
   Position: 10% ✓ Sent
   ...
   Position: 50% ✓ Sent

🔹 Phase 2: Hold at 50% for 2 seconds
   Observe the position...

🔹 Phase 3: Slow return to 0%

   Position: 50% ✓ Sent
   Position: 45% ✓ Sent
   ...
   Position: 0% ✓ Sent

🔹 Phase 4: Stop and disable motor
✅ Test complete for Motor 12

Press Enter to continue with Motor 14...

[Same process for Motor 14]

╔═══════════════════════════════════════════╗
║           Test Complete!                  ║
╚═══════════════════════════════════════════╝

What did you observe?
  Motor 12: Left drive wheel
  Motor 14: Right drive wheel

Document your findings in motor_config.txt

🔒 Ensuring all motors stopped...
✅ Safe shutdown complete
```

---

## 📄 Results File

After testing, `motor_observations.txt` will contain:

```
Test Results Fri Dec  4 14:23:15 PST 2025:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motor 12 (CAN ID 0x0D):
  Function: Left drive wheel
  
Motor 14 (CAN ID 0x0F):
  Function: Right drive wheel
  
Notes:
  - All movements were smooth and controlled
  - Motors returned to starting position successfully
  - No unusual sounds or resistance detected
```

---

## 🎯 Next Steps

After discovering what motors 12 and 14 control:

1. **Update motor config**:
   ```bash
   # Edit motor_config.txt with proper names
   motor_12:
     name: LEFT_DRIVE_WHEEL
     function: Drive
     
   motor_14:
     name: RIGHT_DRIVE_WHEEL
     function: Drive
   ```

2. **Test remaining motors**:
   ```bash
   # The interactive script can test any motor
   sudo ./test_motors_interactive.sh
   # Then select additional motors to test
   ```

3. **Map all motors to brain**:
   ```bash
   ./tools/map_can_motors brain.m
   ```

4. **Integrate with Melvin**:
   ```bash
   sudo ./melvin_motor_runtime brain.m
   ```

---

## 🎓 Understanding Motor Roles

### Common Motor Functions

- **Drive/Locomotion**: Wheels, tracks, legs
- **Manipulation**: Arms, grippers, hands
- **Perception**: Camera pan/tilt, sensor gimbals
- **Interaction**: Head movement, gestures
- **Auxiliary**: Fans, lights, covers

### Example Robot Configuration

```
Motors 0-3:   Wheel drive (4WD mobile base)
Motors 4-7:   Arm joints (shoulder, elbow, wrist, gripper)
Motors 8-9:   Camera pan/tilt
Motors 10-11: Head pan/tilt
Motor 12:     LED ring brightness
Motor 13:     Cooling fan speed
```

---

## 🔐 Safety Reminders

**Before Running Test:**
- [ ] Clear workspace
- [ ] E-stop accessible
- [ ] Person monitoring
- [ ] Robot in safe starting position

**During Test:**
- [ ] Watch robot continuously
- [ ] Ready to press Ctrl+C
- [ ] Listen for unusual sounds
- [ ] Note any resistance or binding

**After Test:**
- [ ] Verify motors stopped
- [ ] Document observations
- [ ] Check for any issues
- [ ] Plan next tests

---

## 📞 Support

If you encounter issues:

1. Check `MOTOR_INTEGRATION.md` for detailed troubleshooting
2. Verify CAN bus configuration
3. Test CAN manually with `candump` and `cansend`
4. Check motor controller documentation

---

**Ready to discover what your motors do?** 🚀

```bash
./deploy_motor_test.sh
```

Then SSH to Jetson and run the interactive test!

