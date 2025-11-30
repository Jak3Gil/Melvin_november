#!/bin/bash
# Fix Melvin service on Jetson - corrects path issues

set -e

echo "=========================================="
echo "Fixing Melvin Service on Jetson"
echo "=========================================="
echo ""

# Configuration
JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "Step 1: Stopping current service..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl stop melvin.service" || echo "Service already stopped"
echo ""

echo "Step 2: Checking Melvin executable..."
EXEC_PATH=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "ls -la /home/melvin/melvin_system/melvin 2>/dev/null | awk '{print \$9}' || echo 'NOT_FOUND'")
if [ "$EXEC_PATH" = "NOT_FOUND" ]; then
    echo "✗ Executable not found at /home/melvin/melvin_system/melvin"
    echo "  Checking if it needs to be built..."
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
        cd /home/melvin/melvin_system
        if [ -f melvin.c ]; then
            echo "Building melvin..."
            gcc -O2 -o melvin melvin.c -ldl -lm -lpthread
            chmod +x melvin
            echo "✓ Built successfully"
        else
            echo "✗ melvin.c not found - need to deploy files first"
        fi
EOF
else
    echo "✓ Executable found"
fi
echo ""

echo "Step 3: Updating service file..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOFSERVICE'
    sudo tee /etc/systemd/system/melvin.service > /dev/null << 'EOF'
[Unit]
Description=Melvin Autonomous AI Runtime
After=network.target

[Service]
Type=simple
User=melvin
WorkingDirectory=/home/melvin/melvin_system
ExecStart=/home/melvin/melvin_system/melvin melvin.m
Restart=always
RestartSec=10
StandardOutput=append:/home/melvin/melvin_system/melvin.log
StandardError=append:/home/melvin/melvin_system/melvin.log

# Resource limits for Jetson (64GB RAM available)
MemoryMax=50G
CPUQuota=800%

[Install]
WantedBy=multi-user.target
EOF
    echo "✓ Service file updated"
EOFSERVICE
echo ""

echo "Step 4: Reloading systemd..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl daemon-reload"
echo "✓ Systemd reloaded"
echo ""

echo "Step 5: Enabling service..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl enable melvin.service"
echo "✓ Service enabled"
echo ""

echo "Step 6: Starting service..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl start melvin.service"
sleep 2
echo ""

echo "Step 7: Checking service status..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl status melvin.service --no-pager -l | head -20"
echo ""

echo "=========================================="
echo "Service Fix Complete!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  Check status: sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'sudo systemctl status melvin.service'"
echo "  View logs:    sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'tail -f /home/melvin/melvin_system/melvin.log'"
echo "  Restart:      sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'sudo systemctl restart melvin.service'"
echo ""

