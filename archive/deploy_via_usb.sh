#!/bin/bash
# Deploy melvin to Jetson via USB (if USB is mounted)

echo "=========================================="
echo "Deploying Melvin via USB"
echo "=========================================="

# Find USB mount
USB_MOUNT=""
for mount in /Volumes/*; do
    if [ -d "$mount" ] && [ -w "$mount" ]; then
        # Check if it looks like a Jetson USB
        if [ -f "$mount/L4T-README" ] || [ -d "$mount/home" ] || [ -d "$mount/melvin" ]; then
            USB_MOUNT="$mount"
            break
        fi
    fi
done

# If no specific Jetson mount, try to find any writable USB
if [ -z "$USB_MOUNT" ]; then
    for mount in /Volumes/*; do
        if [ -d "$mount" ] && [ -w "$mount" ] && [ "$mount" != "/Volumes/Macintosh HD" ]; then
            USB_MOUNT="$mount"
            break
        fi
    done
fi

if [ -z "$USB_MOUNT" ]; then
    echo "ERROR: No USB mount found"
    echo "Please:"
    echo "  1. Plug Jetson USB into Mac"
    echo "  2. Wait for it to mount"
    echo "  3. Run this script again"
    exit 1
fi

echo "Found USB mount: $USB_MOUNT"
echo ""

# Create melvin directory on USB
MELVIN_DIR="$USB_MOUNT/melvin"
mkdir -p "$MELVIN_DIR"
mkdir -p "$MELVIN_DIR/plugins"
mkdir -p "$MELVIN_DIR/ingested_repos"

echo "Copying files to USB..."
cp melvin.c melvin.h "$MELVIN_DIR/"
if [ -f melvin.m ]; then
    cp melvin.m "$MELVIN_DIR/"
    echo "  ✓ melvin.m (brain file)"
fi
echo "  ✓ melvin.c"
echo "  ✓ melvin.h"

# Copy scripts
cp start_melvin.sh monitor_melvin.sh stop_melvin.sh "$MELVIN_DIR/" 2>/dev/null
cp deploy_to_jetson.sh monitor_jetson.sh feed_c_files.sh "$MELVIN_DIR/" 2>/dev/null

# Copy monitor if exists
if [ -f monitor_melvin ]; then
    cp monitor_melvin "$MELVIN_DIR/" 2>/dev/null
fi

# Copy init script if exists
if [ -f init_melvin_simple ]; then
    cp init_melvin_simple "$MELVIN_DIR/" 2>/dev/null
fi

# Create setup script on USB
cat > "$MELVIN_DIR/setup_jetson.sh" << 'EOF'
#!/bin/bash
cd ~/melvin || cd /media/*/melvin || cd /mnt/*/melvin || cd melvin

echo "=========================================="
echo "Setting up Melvin on Jetson"
echo "=========================================="

# Compile melvin
echo "Compiling melvin..."
gcc -o melvin melvin.c -lm -ldl 2>&1 | grep -E "(error|warning)" || echo "✓ Compiled"

# Compile monitor if source exists
if [ -f monitor_melvin.c ]; then
    echo "Compiling monitor..."
    gcc -o monitor_melvin monitor_melvin.c -lm 2>&1 | grep -E "(error|warning)" || echo "✓ Monitor compiled"
fi

# Initialize brain if needed
if [ ! -f melvin.m ]; then
    echo "Initializing brain..."
    if [ -f init_melvin_simple ]; then
        ./init_melvin_simple
    else
        echo "ERROR: No init script found"
        echo "Creating minimal brain..."
        # Create minimal brain manually
        python3 << 'PYEOF'
import struct
import os

# Create minimal brain file
with open('melvin.m', 'wb') as f:
    # Header (256 bytes)
    header = struct.pack('<QQQQQ', 0, 0, 0, 100000, 500000)  # num_nodes, num_edges, tick, node_cap, edge_cap
    header += b'\x00' * (256 - 40)  # Padding
    f.write(header)
    
    # Nodes (100k * 40 bytes = 4MB)
    f.write(b'\x00' * (100000 * 40))
    
    # Edges (500k * 32 bytes = 16MB)
    f.write(b'\x00' * (500000 * 32))
    
print("✓ Created melvin.m")
PYEOF
    fi
fi

# Make scripts executable
chmod +x *.sh 2>/dev/null

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start Melvin:"
echo "  ./start_melvin.sh"
echo ""
echo "To monitor:"
echo "  ./monitor_melvin.sh"
echo ""
EOF

chmod +x "$MELVIN_DIR/setup_jetson.sh"

echo ""
echo "=========================================="
echo "Deployment Complete"
echo "=========================================="
echo ""
echo "Files copied to: $MELVIN_DIR"
echo ""
echo "Next steps:"
echo "  1. Eject USB safely"
echo "  2. Plug into Jetson"
echo "  3. On Jetson, copy files:"
echo "     cp -r /media/*/melvin ~/melvin"
echo "     # or if mounted elsewhere:"
echo "     cp -r /mnt/*/melvin ~/melvin"
echo "  4. Run setup:"
echo "     cd ~/melvin"
echo "     ./setup_jetson.sh"
echo "  5. Start melvin:"
echo "     ./start_melvin.sh"
echo ""
echo "Then monitor from Mac:"
echo "  ./monitor_jetson.sh <jetson_ip>"
echo ""

