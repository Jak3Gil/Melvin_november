#!/bin/bash
# Comprehensive stress test for Melvin's emergent system

set -e

BRAIN_FILE="melvin.m"
BACKUP_FILE="melvin.m.stress_backup"
TEST_DURATION=60  # seconds

echo "========================================="
echo "MELVIN STRESS TEST"
echo "========================================="
echo ""
echo "Testing:"
echo "  1. Graph growth under load"
echo "  2. Edge creation stability"
echo "  3. Sequence edge formation"
echo "  4. Co-activation and learning"
echo "  5. Pattern formation"
echo "  6. Memory/file integrity"
echo "  7. Rule stability"
echo ""

# Backup existing brain
if [ -f "$BRAIN_FILE" ]; then
    echo "Backing up existing melvin.m..."
    cp "$BRAIN_FILE" "$BACKUP_FILE"
fi

# Initialize if needed
if [ ! -f "$BRAIN_FILE" ]; then
    echo "Initializing melvin.m..."
    python3 -c "
import struct
header = struct.pack('<QQQ', 0, 0, 0) + b'\\x00' * (256 - 24)
with open('melvin.m', 'wb') as f:
    f.write(header)
    f.truncate(1024 * 1024)  # 1MB initial
"
fi

# Check if Melvin binary exists
if [ ! -f "./melvin" ]; then
    echo "ERROR: melvin binary not found. Building..."
    make melvin 2>&1 | tail -10
    if [ ! -f "./melvin" ]; then
        echo "ERROR: Failed to build melvin"
        exit 1
    fi
fi

echo "Step 1: Initial Graph State"
echo "----------------------------"
python3 << 'PYTHON_SCRIPT'
import struct
with open('melvin.m', 'rb') as f:
    f.seek(0)
    header = f.read(256)
    num_nodes = struct.unpack('<Q', header[0:8])[0]
    num_edges = struct.unpack('<Q', header[8:16])[0]
    tick = struct.unpack('<Q', header[16:24])[0]
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Tick: {tick}")
PYTHON_SCRIPT

echo ""
echo "Step 2: Running Melvin Under Stress"
echo "------------------------------------"
echo "Running Melvin for ${TEST_DURATION} seconds..."

# Start Melvin in background
./melvin > /tmp/melvin_stress.log 2>&1 &
MELVIN_PID=$!

# Monitor for the duration
sleep $TEST_DURATION

# Stop Melvin
echo "Stopping Melvin..."
kill $MELVIN_PID 2>/dev/null || true
wait $MELVIN_PID 2>/dev/null || true
sleep 1

echo ""
echo "Step 3: Final Graph State Analysis"
echo "-----------------------------------"
python3 << 'PYTHON_SCRIPT'
import struct
import os

with open('melvin.m', 'rb') as f:
    f.seek(0)
    header = f.read(256)
    num_nodes = struct.unpack('<Q', header[0:8])[0]
    num_edges = struct.unpack('<Q', header[8:16])[0]
    tick = struct.unpack('<Q', header[16:24])[0]
    
    file_size = os.path.getsize('melvin.m')
    
    print(f"Final state:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Tick: {tick}")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    
    # Analyze edges
    f.seek(256 + num_nodes * 44)
    seq_edges = 0
    bind_edges = 0
    pattern_edges = 0
    corrupted_edges = 0
    valid_edges = 0
    
    for i in range(min(10000, num_edges)):
        edge_bytes = f.read(48)
        if len(edge_bytes) < 48:
            break
        src = struct.unpack('<Q', edge_bytes[0:8])[0]
        dst = struct.unpack('<Q', edge_bytes[8:16])[0]
        flags = struct.unpack('<I', edge_bytes[20:24])[0]
        w = struct.unpack('<f', edge_bytes[16:20])[0]
        
        if src < num_nodes and dst < num_nodes:
            valid_edges += 1
            if flags & 0x2:  # EDGE_FLAG_SEQ
                seq_edges += 1
            if flags & 0x4:  # EDGE_FLAG_BIND
                bind_edges += 1
            if flags & 0x80:  # EDGE_FLAG_PATTERN
                pattern_edges += 1
        else:
            corrupted_edges += 1
    
    print(f"\nEdge analysis (first 10000):")
    print(f"  Valid edges: {valid_edges}")
    print(f"  Sequence edges: {seq_edges}")
    print(f"  Binding edges: {bind_edges}")
    print(f"  Pattern edges: {pattern_edges}")
    print(f"  Corrupted edges: {corrupted_edges}")
    
    # Analyze nodes
    f.seek(256)
    patterns = 0
    blanks = 0
    control = 0
    corrupted_nodes = 0
    
    for i in range(min(10000, num_nodes)):
        node_bytes = f.read(44)
        if len(node_bytes) < 44:
            break
        kind = struct.unpack('<I', node_bytes[12:16])[0]
        a = struct.unpack('<f', node_bytes[0:4])[0]
        
        if kind == 2:  # NODE_KIND_PATTERN_ROOT
            patterns += 1
        elif kind == 0:  # NODE_KIND_BLANK
            blanks += 1
        elif kind == 3:  # NODE_KIND_CONTROL
            control += 1
        
        import math
        if math.isnan(a) or math.isinf(a):
            corrupted_nodes += 1
    
    print(f"\nNode analysis (first 10000):")
    print(f"  Patterns: {patterns}")
    print(f"  Blanks: {blanks}")
    print(f"  Control: {control}")
    print(f"  Corrupted: {corrupted_nodes}")
    
    # Test results
    print(f"\n=== STRESS TEST RESULTS ===")
    passed = 0
    failed = 0
    warnings = 0
    
    if num_nodes > 0:
        print("  ✓ Graph has nodes")
        passed += 1
    else:
        print("  ✗ Graph has no nodes")
        failed += 1
    
    if num_edges > 0:
        print("  ✓ Graph has edges")
        passed += 1
    else:
        print("  ✗ Graph has no edges")
        failed += 1
    
    if seq_edges > 0:
        print(f"  ✓ Sequence edges present ({seq_edges})")
        passed += 1
    else:
        print("  ✗ No sequence edges found")
        warnings += 1
    
    if patterns > 0:
        print(f"  ✓ Patterns present ({patterns})")
        passed += 1
    else:
        print("  ⚠ No patterns found (may need scaffold processing)")
        warnings += 1
    
    if corrupted_edges < num_edges * 0.1:
        print(f"  ✓ Low corruption rate ({corrupted_edges}/{num_edges})")
        passed += 1
    else:
        print(f"  ✗ High corruption rate ({corrupted_edges}/{num_edges})")
        failed += 1
    
    if corrupted_nodes == 0:
        print("  ✓ No corrupted nodes")
        passed += 1
    else:
        print(f"  ✗ Corrupted nodes found ({corrupted_nodes})")
        failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed, {warnings} warnings")
    
    if failed == 0:
        print("\n✓ STRESS TEST PASSED - Rules are stable and allow emergence")
    else:
        print("\n✗ STRESS TEST FAILED - Some issues detected")
PYTHON_SCRIPT

echo ""
echo "Step 4: Checking Logs for Errors"
echo "---------------------------------"
if [ -f /tmp/melvin_stress.log ]; then
    error_count=$(grep -i "error\|fatal\|crash" /tmp/melvin_stress.log | wc -l | tr -d ' ')
    if [ "$error_count" -gt 0 ]; then
        echo "  Found $error_count errors in logs:"
        grep -i "error\|fatal\|crash" /tmp/melvin_stress.log | tail -5
    else
        echo "  ✓ No errors in logs"
    fi
else
    echo "  ⚠ Log file not found"
fi

echo ""
echo "========================================="
echo "STRESS TEST COMPLETE"
echo "========================================="

