#!/bin/bash
# Fresh setup script for Jetson after OS reinstall
# Run this after you've reinstalled JetPack/Jetson OS

set -e

echo "=== Fresh Melvin Setup on Jetson ==="
echo ""

# Configuration
JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "Step 1: Installing dependencies..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    sudo apt update
    sudo apt upgrade -y
    
    # Essential build tools
    sudo apt install -y build-essential git curl wget
    
    # Compiler and build tools
    sudo apt install -y gcc clang make cmake
    
    # Libraries
    sudo apt install -y libc6-dev libdl-dev
    
    # Utilities
    sudo apt install -y sshpass
    
    echo "✅ Dependencies installed"
EOF

echo ""
echo "Step 2: Creating Melvin directories..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    mkdir -p /home/melvin/melvin_system
    mkdir -p /home/melvin/scaffolds
    mkdir -p /home/melvin/MELVIN/bin
    mkdir -p /home/melvin/MELVIN/plugins
    
    echo "✅ Directories created"
EOF

echo ""
echo "Step 3: Deploying Melvin core files..."
scp melvin.c melvin.h "$JETSON_HOST:/home/melvin/melvin_system/"
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" "cd /home/melvin/melvin_system && chmod +x melvin.c"
echo "✅ Core files deployed"

echo ""
echo "Step 4: Building Melvin..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    cd /home/melvin/melvin_system
    
    # Compile melvin.c
    gcc -O2 -o melvin_jetson melvin.c -ldl -lm -lpthread
    
    echo "✅ Melvin compiled"
EOF

echo ""
echo "Step 5: Initializing melvin.m (or restore backup)..."
echo "  If you have a backup, restore it now!"
echo "  Otherwise, we'll initialize fresh melvin.m"
echo ""
echo "Step 6: Setting up display output..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    # Create display output program (we created this earlier)
    # Will be deployed separately
    
    echo "✅ Display setup ready"
EOF

echo ""
echo "Step 7: Setting up auto-start..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    # Create systemd service for Melvin
    sudo tee /etc/systemd/system/melvin.service > /dev/null << 'EOFSERVICE'
[Unit]
Description=Melvin AI System
After=network.target

[Service]
Type=simple
User=melvin
WorkingDirectory=/home/melvin/melvin_system
ExecStart=/home/melvin/melvin_system/melvin_jetson melvin.m
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOFSERVICE
    
    sudo systemctl daemon-reload
    sudo systemctl enable melvin.service
    
    echo "✅ Auto-start configured"
EOF

echo ""
echo "✅ Fresh setup complete!"
echo ""
echo "Next steps:"
echo "  1. Restore melvin.m backup if you have one"
echo "  2. Deploy display output program"
echo "  3. Start Melvin: sudo systemctl start melvin"
echo "  4. Check display: should show graph stats"

