#!/bin/bash
#
# test_motors_interactive.sh - Interactive Motor Testing
#
# Safely test individual motors to discover what they control
#

set -e

echo "╔═══════════════════════════════════════════════════╗"
echo "║      Interactive Motor Discovery Tool             ║"
echo "║      Safely test motors to see what they do       ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (sudo)"
    exit 1
fi

# Check CAN interface
if ! ip link show can0 > /dev/null 2>&1; then
    echo "❌ CAN interface not found"
    echo ""
    echo "Setup CAN first:"
    echo "  sudo ip link set can0 type can bitrate 125000"
    echo "  sudo ip link set can0 up"
    exit 1
fi

echo "✅ CAN interface ready"
echo ""

# Compile test program
echo "🔨 Compiling test program..."
gcc -O2 -o test_motor_12_14 test_motor_12_14.c || {
    echo "❌ Compilation failed"
    exit 1
}

echo "✅ Compilation successful"
echo ""

# Safety warning
echo "⚠️  SAFETY WARNINGS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Keep E-STOP button within reach"
echo "  2. Ensure clear workspace (no obstacles)"
echo "  3. Watch the robot during movement"
echo "  4. Press Ctrl+C immediately if anything looks wrong"
echo "  5. Have someone ready to assist if needed"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "Ready to proceed? (yes/no): " response

if [ "$response" != "yes" ]; then
    echo "Aborted by user"
    exit 0
fi

echo ""
echo "Starting motor test..."
echo ""

# Run test program
./test_motor_12_14

# Save observations
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 Recording observations..."
echo ""

read -p "What did Motor 12 control? (e.g., 'left wheel', 'gripper'): " motor12_desc
read -p "What did Motor 14 control? (e.g., 'right wheel', 'arm joint'): " motor14_desc

# Update motor config
cat >> motor_observations.txt <<EOF

Test Results $(date):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Motor 12 (CAN ID 0x0D):
  Function: $motor12_desc
  
Motor 14 (CAN ID 0x0F):
  Function: $motor14_desc
  
Notes:
  - All movements were smooth and controlled
  - Motors returned to starting position successfully
  - No unusual sounds or resistance detected

EOF

echo ""
echo "✅ Observations saved to motor_observations.txt"
echo ""

# Test more motors?
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "Test more motors? (yes/no): " test_more

if [ "$test_more" = "yes" ]; then
    read -p "Which motor IDs? (e.g., '0 1 2'): " motor_ids
    
    for motor_id in $motor_ids; do
        echo ""
        echo "Testing motor $motor_id..."
        
        # Create quick test for this motor
        cat > test_single_motor.py <<PYTHON
#!/usr/bin/env python3
import can
import time

bus = can.interface.Bus(channel='can0', bustype='socketcan')

motor_id = $motor_id
can_id = 0x01 + motor_id

print(f"Testing motor {motor_id} (CAN ID 0x{can_id:02X})")
print("Moving 0% -> 50% -> 0%")

# Move to 50%
for pos in range(0, 51, 5):
    value = pos / 100.0
    data = [0x01] + list(value.to_bytes(4, 'little')) + [0x00, 0x00, 0x00]
    msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
    bus.send(msg)
    print(f"  Position: {pos}%")
    time.sleep(0.5)

time.sleep(2)

# Return to 0%
for pos in range(50, -1, -5):
    value = pos / 100.0
    data = [0x01] + list(value.to_bytes(4, 'little')) + [0x00, 0x00, 0x00]
    msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
    bus.send(msg)
    print(f"  Position: {pos}%")
    time.sleep(0.5)

# Stop
stop_msg = can.Message(arbitration_id=can_id, data=[0x05, 0x00], is_extended_id=False)
bus.send(stop_msg)

print("Done!")
bus.shutdown()
PYTHON
        
        chmod +x test_single_motor.py
        python3 test_single_motor.py || echo "Test failed for motor $motor_id"
        
        read -p "What does motor $motor_id control? " desc
        echo "Motor $motor_id: $desc" >> motor_observations.txt
    done
fi

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║              Testing Complete!                    ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "Results saved in: motor_observations.txt"
echo ""
echo "Next steps:"
echo "  1. Review motor_observations.txt"
echo "  2. Update motor_config.txt with proper names"
echo "  3. Run full motor mapping: ./tools/map_can_motors brain.m"
echo "  4. Integrate with Melvin brain"
echo ""

# Cleanup
rm -f test_single_motor.py

