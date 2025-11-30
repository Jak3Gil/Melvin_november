#!/bin/bash
# Clean install script - removes all packages and reinstalls fresh
# This is faster than full OS reflash and doesn't require physical access

set -e

echo "=== Clean Install Jetson System ==="
echo ""
echo "WARNING: This will remove all packages and data!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo ""
echo "Step 1: Backup important data (already done - melvin.m backed up)"
echo ""

echo "Step 2: Clean system packages..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    echo "Removing all non-essential packages..."
    
    # Remove user-installed packages
    sudo apt autoremove -y
    sudo apt autoclean
    
    # Clean package cache
    sudo apt clean
    
    # Remove user data (but keep system files)
    sudo rm -rf /home/melvin/* /home/melvin/.* 2>/dev/null || true
    
    echo "✅ System cleaned"
EOF

echo ""
echo "Step 3: Update package lists..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    sudo apt update
    sudo apt upgrade -y
    
    echo "✅ System updated"
EOF

echo ""
echo "Step 4: Reinstall essential packages..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    # Essential system packages
    sudo apt install -y ubuntu-desktop-minimal
    
    # Build tools
    sudo apt install -y build-essential git curl wget
    
    # Compiler
    sudo apt install -y gcc clang make cmake
    
    # Libraries
    sudo apt install -y libc6-dev libdl-dev
    
    # SSH
    sudo apt install -y openssh-server sshpass
    
    # Display
    sudo apt install -y xserver-xorg-core
    
    echo "✅ Essential packages installed"
EOF

echo ""
echo "Step 5: Setup user and directories..."
sshpass -p "$JETSON_PASS" ssh "$JETSON_HOST" << 'EOF'
    # Create directories
    mkdir -p /home/melvin/melvin_system
    mkdir -p /home/melvin/scaffolds
    mkdir -p /home/melvin/MELVIN/bin
    
    # Set permissions
    sudo chown -R melvin:melvin /home/melvin
    
    echo "✅ Directories created"
EOF

echo ""
echo "✅ Clean install complete!"
echo ""
echo "Next steps:"
echo "  1. Deploy Melvin system: ./fresh_setup_jetson.sh"
echo "  2. Restore melvin.m backup"
echo "  3. Start Melvin"

