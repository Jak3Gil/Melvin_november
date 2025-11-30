#!/bin/bash
# Commands to run in VM shell to set up Melvin

echo "=========================================="
echo "Commands to Run in Your VM Shell"
echo "=========================================="
echo ""
echo "STEP 1: Install build tools"
echo "----------------------------------------"
echo "sudo apt update"
echo "sudo apt install -y build-essential gcc make git openssh-server"
echo ""
echo "STEP 2: Start SSH (for file transfer)"
echo "----------------------------------------"
echo "sudo systemctl start ssh"
echo "sudo systemctl enable ssh"
echo ""
echo "STEP 3: Check your VM IP"
echo "----------------------------------------"
echo "ip addr show | grep 'inet ' | grep -v '127.0.0.1'"
echo ""
echo "STEP 4: Create Melvin directory"
echo "----------------------------------------"
echo "mkdir -p ~/melvin_november"
echo ""
echo "After SSH is running, from Mac run:"
echo "  ./transfer_and_test.sh 192.168.64.2"
echo ""
echo "=========================================="




