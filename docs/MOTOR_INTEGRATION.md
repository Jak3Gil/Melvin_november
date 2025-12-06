# Motor Integration Guide

**Integrating 14 CAN Motors with Melvin Brain**

---

## 🎯 Overview

This guide explains how to integrate 14 motors connected via USB-to-CAN adapter with the Melvin brain, enabling **learned motor control** through teachable EXEC nodes.

### Key Innovation

**Traditional Robot**: Hardcoded motor control
```c
if (sensor_input) {
    motor_move(0, position);  // Hardcoded!
}
```

**Melvin Robot**: Learned motor control
```
Pattern discovered → Routes to EXEC node → ARM64 code executes → Motor moves
Everything learned through experience! ✨
```

---

## 🔧 Hardware Setup

### Components

1. **Jetson AGX Orin** (or other Jetson device)
2. **USB-to-CAN Adapter** (e.g., PEAK PCAN-USB, CANable)
3. **14 Motors** with CAN interface (each with unique CAN ID)
4. **CAN Bus Wiring** (CAN-H, CAN-L, 120Ω termination)

### Physical Connections

```
Jetson USB Port
    │
    ├─── USB-to-CAN Adapter
           │
           ├─── CAN Bus (twisted pair)
                 │
                 ├─── Motor 0 (CAN ID: 0x01)
                 ├─── Motor 1 (CAN ID: 0x02)
                 ├─── Motor 2 (CAN ID: 0x03)
                 │    ...
                 └─── Motor 13 (CAN ID: 0x0E)
```

### CAN Termination

- Place 120Ω resistors at **both ends** of CAN bus
- Maximum cable length: ~40m at 1 Mbps, ~1000m at 50 kbps

---

## 🚀 Quick Start

### 1. Setup (One-Time)

```bash
# On Jetson
cd /home/melvin/Melvin_november

# Run setup script
sudo ./setup_jetson_motors.sh brain_teachable.m
```

This will:
- ✅ Install CAN utilities
- ✅ Configure CAN interface (can0)
- ✅ Scan for motors
- ✅ Map motors to brain ports
- ✅ Teach motor control code to EXEC nodes
- ✅ Create systemd service for auto-start

### 2. Test Motors

```bash
# Test all motors
sudo ./test_motor_exec brain_teachable.m all

# Test specific motor
sudo ./test_motor_exec brain_teachable.m 0

# Run sequence on motor
sudo ./test_motor_exec brain_teachable.m 0 seq

# Test pattern routing
sudo ./test_motor_exec brain_teachable.m 0 pattern
```

### 3. Run Motor Runtime

```bash
# Manual start
sudo ./melvin_motor_runtime brain_teachable.m

# Or enable auto-start
sudo systemctl enable melvin-motors
sudo systemctl start melvin-motors
```

---

## 🧠 Architecture

### Port Mapping

| Component | Port Range | Purpose |
|-----------|-----------|---------|
| Motor Feedback | 200-213 | Motor state input (position, velocity, torque) |
| Motor EXEC | 2200-2213 | EXEC nodes with motor control code |
| Motor Commands | 3100-3113 | Motor command output ports |

### Data Flow

```
┌─────────────────────────────────────────────┐
│ SENSORS (Camera, Mic)                       │
└──────────────┬──────────────────────────────┘
               │ Raw data + AI labels
               ↓
┌─────────────────────────────────────────────┐
│ MELVIN BRAIN (brain.m)                      │
│                                              │
│  Patterns discovered:                       │
│    "PERSON_APPROACHING" (from sensors)      │
│         ↓                                    │
│  Routes to:                                  │
│    EXEC_2200 (wave_hand)                    │
│         ↓                                    │
│  Executes:                                   │
│    ARM64 motor control code                 │
│         ↓                                    │
│  Activates:                                  │
│    Port 3100 (motor 0 command)              │
└──────────────┬──────────────────────────────┘
               │ Command value
               ↓
┌─────────────────────────────────────────────┐
│ MOTOR RUNTIME (melvin_motor_runtime)        │
│  - Monitors EXEC activations                │
│  - Sends CAN commands                       │
│  - Reads motor feedback                     │
│  - Feeds state back to brain                │
└──────────────┬──────────────────────────────┘
               │ CAN messages
               ↓
┌─────────────────────────────────────────────┐
│ PHYSICAL MOTORS                              │
│  Motor 0: Wrist rotation                    │
│  Motor 1: Elbow joint                       │
│  Motor 2: Shoulder                          │
│  ...                                         │
└──────────────────────────────────────────────┘
```

---

## 🎓 How Learning Works

### Phase 1: Preseeding (Bootstrap)

The `map_can_motors` tool:

1. **Scans CAN bus** for motors
2. **Creates EXEC nodes** with ARM64 motor control code
3. **Bootstraps weak edges** (like reflexes)

```c
// Example: Weak reflex edge
pattern_wave → motor_0_exec (strength: 0.2)
```

### Phase 2: Experience (Learning)

```
T=0s:  Person waves at robot
       → Camera: "HAND_DETECTED"
       → Mic: "HELLO_AUDIO"
       → Pattern forms: [HAND, HELLO]

T=5s:  Brain discovers: greeting → should wave back
       → Weak edge activates: [GREETING] → motor_0_exec
       → Motor 0 moves (tentative wave)
       
T=10s: Person smiles (positive feedback)
       → Edge strengthens: 0.2 → 0.4
       → Brain learns: waving is good response

T=60s: After many interactions
       → Edge very strong: 0.8
       → Automatic waving response!
       → Brain learned social behavior!
```

### Phase 3: Hierarchical Composition

Brain composes complex behaviors:

```
Low-level patterns:
  [PERSON_DETECTED] + [GREETING_HEARD]
       ↓
Mid-level composition:
  [SOCIAL_INTERACTION]
       ↓
High-level behavior:
  Routes to: wave_exec + speak_exec
       ↓
Coordinated response:
  Waves AND says "Hello!"
```

**None of this is hardcoded!** All emerges from:
- Pattern discovery (co-activation)
- Hierarchical composition
- Learned routing to EXEC nodes
- Feedback-driven edge strengthening

---

## 🔬 Motor Control Code

### ARM64 Machine Code

The EXEC nodes contain actual ARM64 machine code that:

1. Prepares CAN frame
2. Makes syscall to write to CAN socket
3. Returns status

**Simplified example** (actual code in `tools/map_can_motors.c`):

```asm
; ARM64 motor control function
; x0 = motor_id, x1 = command, x2 = value

stp x29, x30, [sp, #-16]!    ; Prologue
mov x29, sp

; Prepare CAN frame
str x0, [sp, #24]             ; can_id
str x1, [sp, #32]             ; command  
str x2, [sp, #40]             ; data

; Syscall: write(socket, frame, size)
mov x0, #3                    ; socket fd
mov x1, sp                    ; frame pointer
mov x2, #16                   ; frame size
mov x8, #64                   ; SYS_write
svc #0                        ; Make syscall

; Check return
cmp w0, #0
cset w0, gt                   ; Return 1 if success

ldp x29, x30, [sp], #16       ; Epilogue
ret
```

### Why Machine Code?

**Benefits**:
- **Self-contained**: Brain has all code, no external dependencies
- **Portable**: Can run on any ARM64 system
- **Fast**: Direct execution, no interpretation
- **Teachable**: Brain can learn which code to execute when

---

## 🛠️ Tools Reference

### map_can_motors

Maps motors to brain and teaches control code.

```bash
sudo ./tools/map_can_motors brain.m
```

**What it does**:
1. Scans CAN bus (IDs 0x01-0x0E)
2. Creates EXEC node for each motor
3. Stores ARM64 control code in blob
4. Creates port patterns
5. Saves `motor_config.txt`

**Output**:
```
Motor 0: CAN 0x01 → EXEC 2200 → Port 3100
Motor 1: CAN 0x02 → EXEC 2201 → Port 3101
...
```

### test_motor_exec

Tests motor control through EXEC nodes.

```bash
# Test all motors
sudo ./test_motor_exec brain.m all

# Test motor 0
sudo ./test_motor_exec brain.m 0

# Run movement sequence
sudo ./test_motor_exec brain.m 0 seq

# Test pattern routing
sudo ./test_motor_exec brain.m 0 pattern
```

**Pattern test** feeds pattern and checks if it routes to motor EXEC.

### melvin_motor_runtime

Real-time motor control runtime.

```bash
sudo ./melvin_motor_runtime brain.m [motor_config.txt]
```

**Runs continuously**:
- Monitors EXEC nodes (2200-2213)
- Sends CAN commands when activated
- Reads motor feedback
- Feeds state back to brain (ports 200-213)
- Updates at 1kHz

**Stop with**: Ctrl+C (safely stops all motors)

---

## 📋 Motor Configuration File

Generated by `map_can_motors`:

```yaml
# motor_config.txt

motor_0:
  name: MOTOR_0
  can_id: 0x01
  port_id: 3100
  exec_id: 2200
  feedback_id: 200

motor_1:
  name: MOTOR_1
  can_id: 0x02
  port_id: 3101
  exec_id: 2201
  feedback_id: 201

# ... etc for all 14 motors
```

---

## 🧪 Testing & Validation

### Basic Connectivity Test

```bash
# Check CAN interface
ip link show can0

# Monitor CAN traffic
candump can0

# Send test frame
cansend can0 123#DEADBEEF
```

### Motor Discovery Test

```bash
# Scan for motors
sudo ./tools/map_can_motors brain.m
```

Expected output:
```
✅ Motor 0 detected (CAN ID 0x01)
✅ Motor 1 detected (CAN ID 0x02)
...
Found 14 motors
```

### EXEC Execution Test

```bash
# Test that EXEC nodes can control motors
sudo ./test_motor_exec brain.m all
```

Expected output:
```
Motor 0: ✅ OK
Motor 1: ✅ OK
...
✅ Tested 14 motors
```

### Pattern Routing Test

```bash
# Test learned routing
sudo ./test_motor_exec brain.m 0 pattern
```

Expected output:
```
Feeding pattern: MOVE_MOTOR_0
Propagating through graph...
EXEC node activation: 0.8234
✅ Pattern successfully routed to motor EXEC!
```

### Full Integration Test

```bash
# Run motor runtime with full hardware integration
sudo ./melvin_motor_runtime brain.m &

# In another terminal, run sensor integration
python3 multimodal_brain_integration.py brain.m

# Brain should now learn to control motors based on sensory input!
```

---

## 🐛 Troubleshooting

### CAN Interface Not Found

```bash
# Check USB-to-CAN adapter is connected
lsusb

# Try manual setup
sudo slcand -o -c -s6 /dev/ttyUSB0 can0
sudo ip link set can0 up
```

### No Motors Detected

1. Check power to motors
2. Verify CAN wiring (CAN-H, CAN-L)
3. Check termination resistors (120Ω at both ends)
4. Try manual CAN send:
   ```bash
   cansend can0 001#10  # Query motor 1
   candump can0         # Listen for response
   ```

### Motors Don't Respond

1. Check `motor_config.txt` has correct CAN IDs
2. Verify EXEC nodes have code:
   ```bash
   ./tools/inspect_graph brain.m | grep "EXEC.*MOTOR"
   ```
3. Check motor runtime is actually sending:
   ```bash
   # In one terminal
   sudo ./melvin_motor_runtime brain.m
   
   # In another
   candump can0  # Should see CAN frames when EXEC activates
   ```

### Permission Denied

CAN requires root access:
```bash
# Either run as root
sudo ./melvin_motor_runtime brain.m

# Or add user to dialout group
sudo usermod -a -G dialout $USER
# Then logout and login
```

---

## 🔒 Safety Considerations

### Motor Limits

Always implement:
- **Position limits**: Software and hardware
- **Velocity limits**: Prevent sudden movements
- **Torque limits**: Protect motors and structures
- **Emergency stop**: Hardware E-stop button

### Testing Procedure

1. **Bench test first**: Test without load
2. **Slow speeds**: Start with low velocities
3. **Limited range**: Restrict motion initially
4. **Supervision**: Always supervise during learning
5. **Kill switch**: Keep E-stop accessible

### Safe Shutdown

The motor runtime handles graceful shutdown:

```c
// On Ctrl+C or SIGTERM:
1. Stop all motors (send zero command)
2. Close CAN socket
3. Save brain state
4. Exit
```

---

## 🎯 Example Applications

### 1. Robotic Arm

```
Motors 0-5: Arm joints (shoulder, elbow, wrist)
Motors 6-7: Gripper (open/close)

Brain learns:
  "PICK_UP_OBJECT" → Coordinate motors 0-7
  "WAVE_HAND" → Move motors 3-5
  "POINT_AT_OBJECT" → Extend arm toward object
```

### 2. Mobile Robot

```
Motors 0-3: Wheel motors (differential drive)
Motors 4-5: Pan/tilt for camera

Brain learns:
  "MOVE_FORWARD" → Motors 0-3 same speed
  "TURN_LEFT" → Left motors slower
  "TRACK_FACE" → Motors 4-5 follow person
```

### 3. Humanoid Robot

```
Motors 0-13: Various joints

Brain learns:
  Social interactions (wave, nod, gesture)
  Balance and walking
  Manipulation (grasp, place)
  
All learned from experience, not programmed!
```

---

## 🚀 Advanced Features

### Multi-Modal Learning

Combine motor control with other senses:

```python
# Python orchestrator
while True:
    # Sensors
    audio = capture_audio()
    video = capture_video()
    
    # AI preprocessing
    speech = whisper(audio)
    objects = mobilenet(video)
    
    # Feed to brain
    feed_string(brain, speech)
    feed_string(brain, objects)
    
    # Brain learns patterns
    melvin_call_entry(brain)
    
    # Check if brain wants to move motors
    for motor_id in range(14):
        if exec_activated(brain, motor_id):
            # Motor runtime handles execution
            pass
    
    # Brain learns: speech + vision → motor control
```

### Hierarchical Motor Control

Brain composes primitive movements into complex behaviors:

```
Level 1: Joint control
  move_joint_0, move_joint_1, ...

Level 2: Limb control (compose joints)
  move_arm, move_leg, move_hand

Level 3: Actions (compose limbs)
  reach_for_object, wave_hello, walk_forward

Level 4: Behaviors (compose actions)
  greet_person, pick_and_place, follow_human
```

### Reinforcement Learning

Brain learns through trial and error:

```
Action: Wave at person
Feedback: Person waves back
→ Strengthen edge: greeting → wave

Action: Wave at wall
Feedback: No response
→ Weaken edge: object → wave

After many trials:
  Strong: person → wave
  Weak: object → wave
  
Brain learned social context!
```

---

## 📊 Performance Metrics

### Timing

- **CAN bus**: Up to 1 Mbps (typically 125-500 kbps)
- **Runtime loop**: 1 kHz (1ms)
- **EXEC activation**: < 10ms
- **Motor command latency**: < 20ms total

### Scalability

- **14 motors**: Tested and working
- **More motors**: Extend port ranges
- **Multiple CAN buses**: Use can0, can1, etc.

### Learning Rate

Depends on:
- Frequency of patterns
- Clarity of feedback
- Edge update parameters

Typical learning times:
- Simple reflex: Minutes
- Complex behavior: Hours
- Social intelligence: Days

---

## 🎓 Next Steps

### After Setup

1. **Baseline testing**: Verify all motors respond
2. **Pattern seeding**: Feed common movement patterns
3. **Supervised learning**: Provide feedback on actions
4. **Autonomous learning**: Let brain explore and learn

### Integration with Full System

```bash
# Start all components

# 1. Motor runtime
sudo ./melvin_motor_runtime brain.m &

# 2. Sensor integration
python3 multimodal_brain_integration.py brain.m &

# 3. Monitor progress
./monitor_melvin.sh
```

### Monitoring Learning

```bash
# Watch edge strengths
watch -n 1 './tools/inspect_graph brain.m | grep MOTOR'

# Watch motor activity
candump can0

# Check pattern formation
tail -f /tmp/melvin_learning.log
```

---

## 🔗 Related Documentation

- `HARDWARE_INTEGRATION_REFINED.md` - Overall hardware architecture
- `TEACHABLE_EXEC_VISION.md` - EXEC node system
- `HIERARCHICAL_COMPOSITION_DESIGN.md` - Pattern composition
- `DEPLOYMENT_READINESS.md` - Full system deployment

---

## 💡 Key Takeaways

1. **Motors are output ports** (3100-3113)
2. **EXEC nodes contain ARM64 code** (2200-2213)
3. **Brain learns routing** (pattern → EXEC → motor)
4. **Everything is learned**, not hardcoded
5. **Hierarchical composition** enables complex behaviors
6. **Real-time feedback** drives learning

**The brain discovers how to control motors through experience!** ✨

---

## 🤝 Contributing

Found an issue or have an improvement?

1. Test your change
2. Document behavior
3. Submit findings

---

**Ready to teach your robot to move? Let's go!** 🚀

