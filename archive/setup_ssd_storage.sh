#!/bin/bash
# Set up 4TB SSD for Melvin data storage

set -e

echo "=========================================="
echo "Setting up 4TB SSD for Melvin Storage"
echo "=========================================="
echo ""

# Configuration
JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "Step 1: Checking SSD status..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    echo "=== SSD Information ==="
    lsblk | grep nvme
    echo ""
    echo "=== Mount Status ==="
    df -h | grep nvme
    echo ""
    echo "=== Current Usage ==="
    du -sh /mnt/melvin_ssd/* 2>/dev/null | head -10 || echo "SSD is empty"
EOF
echo ""

echo "Step 2: Creating Melvin directories on SSD..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    sudo mkdir -p /mnt/melvin_ssd/melvin_data
    sudo mkdir -p /mnt/melvin_ssd/melvin_backups
    sudo mkdir -p /mnt/melvin_ssd/melvin_logs
    sudo mkdir -p /mnt/melvin_ssd/melvin_graphs
    
    # Set ownership
    sudo chown -R melvin:melvin /mnt/melvin_ssd/melvin_*
    
    echo "✓ Directories created"
    ls -la /mnt/melvin_ssd/ | grep melvin
EOF
echo ""

echo "Step 3: Setting up auto-mount (if not already mounted)..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    # Check if already in fstab
    if ! grep -q "/mnt/melvin_ssd" /etc/fstab; then
        echo "Adding to /etc/fstab for auto-mount..."
        # Get UUID
        UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1p2)
        if [ -n "$UUID" ]; then
            echo "UUID=$UUID /mnt/melvin_ssd ntfs-3g defaults,uid=1000,gid=1000,umask=022 0 2" | sudo tee -a /etc/fstab
            echo "✓ Added to fstab"
        else
            echo "⚠️  Could not get UUID, manual mount may be needed"
        fi
    else
        echo "✓ Already in fstab"
    fi
EOF
echo ""

echo "Step 4: Creating symlinks for easy access..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    # Create symlinks in home directory
    ln -sf /mnt/melvin_ssd/melvin_data ~/melvin_data
    ln -sf /mnt/melvin_ssd/melvin_backups ~/melvin_backups
    ln -sf /mnt/melvin_ssd/melvin_logs ~/melvin_logs
    ln -sf /mnt/melvin_ssd/melvin_graphs ~/melvin_graphs
    
    echo "✓ Symlinks created"
    ls -la ~/ | grep melvin
EOF
echo ""

echo "Step 5: Testing write access..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    echo "Testing SSD write..." > /mnt/melvin_ssd/melvin_data/test.txt
    if [ -f /mnt/melvin_ssd/melvin_data/test.txt ]; then
        echo "✓ Write test successful"
        rm /mnt/melvin_ssd/melvin_data/test.txt
    else
        echo "✗ Write test failed"
    fi
EOF
echo ""

echo "Step 6: Checking available space..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "df -h /mnt/melvin_ssd"
echo ""

echo "=========================================="
echo "SSD Setup Complete!"
echo "=========================================="
echo ""
echo "SSD Storage Structure:"
echo "  /mnt/melvin_ssd/melvin_data/     - Main data storage"
echo "  /mnt/melvin_ssd/melvin_backups/  - Backup files"
echo "  /mnt/melvin_ssd/melvin_logs/     - Log files"
echo "  /mnt/melvin_ssd/melvin_graphs/    - Graph exports"
echo ""
echo "Symlinks in home directory:"
echo "  ~/melvin_data -> /mnt/melvin_ssd/melvin_data"
echo "  ~/melvin_backups -> /mnt/melvin_ssd/melvin_backups"
echo "  ~/melvin_logs -> /mnt/melvin_ssd/melvin_logs"
echo "  ~/melvin_graphs -> /mnt/melvin_ssd/melvin_graphs"
echo ""
echo "To move melvin.m to SSD:"
echo "  sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'mv ~/melvin_system/melvin.m ~/melvin_backups/melvin.m.$(date +%Y%m%d)'"
echo "  (Then update service to point to new location)"
echo ""

