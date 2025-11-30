#!/bin/bash
# Set up SSH keys for VM access

set -e

VM_USER="${2:-melvin}"
VM_HOST="${1:-192.168.64.2}"
VM_SSH_PORT="${3:-22}"

echo "=========================================="
echo "Setting up SSH Keys for VM Access"
echo "=========================================="
echo ""
echo "VM: $VM_USER@$VM_HOST:$VM_SSH_PORT"
echo ""

# Check if SSH key exists
SSH_KEY="$HOME/.ssh/id_rsa"
SSH_PUBKEY="$HOME/.ssh/id_rsa.pub"

if [ ! -f "$SSH_KEY" ]; then
    echo "Generating SSH key pair..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY" -N "" -C "melvin-vm-access"
    echo "✓ SSH key generated"
else
    echo "✓ SSH key already exists: $SSH_KEY"
fi

echo ""
echo "Public key to install on VM:"
echo "----------------------------------------"
cat "$SSH_PUBKEY"
echo "----------------------------------------"
echo ""

# Try to copy key using ssh-copy-id (if password auth works)
echo "Attempting to copy key to VM..."
echo "(You may need to enter password once)"
echo ""

if ssh-copy-id -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" 2>&1; then
    echo ""
    echo "✓✓✓ SSH key installed successfully! ✓✓✓"
    echo ""
    echo "Testing connection..."
    if ssh -p "$VM_SSH_PORT" "$VM_USER@$VM_HOST" "echo 'SSH key authentication works!'"; then
        echo ""
        echo "=========================================="
        echo "✓✓✓ SSH SETUP COMPLETE ✓✓✓"
        echo "=========================================="
        echo ""
        echo "You can now run:"
        echo "  ./run_stress_test_on_vm.sh $VM_HOST"
        echo ""
        exit 0
    fi
else
    echo ""
    echo "⚠️  Automatic key copy failed (password auth may be disabled)"
    echo ""
    echo "=========================================="
    echo "MANUAL SETUP REQUIRED"
    echo "=========================================="
    echo ""
    echo "Option 1: Enable password auth temporarily in VM"
    echo "  Inside VM, run:"
    echo "    sudo nano /etc/ssh/sshd_config"
    echo "    # Change: PasswordAuthentication yes"
    echo "    sudo systemctl restart ssh"
    echo "  Then run this script again"
    echo ""
    echo "Option 2: Manually copy the key"
    echo "  Copy this public key:"
    echo "----------------------------------------"
    cat "$SSH_PUBKEY"
    echo "----------------------------------------"
    echo ""
    echo "  Then in VM, run:"
    echo "    mkdir -p ~/.ssh"
    echo "    chmod 700 ~/.ssh"
    echo "    echo 'PASTE_PUBLIC_KEY_HERE' >> ~/.ssh/authorized_keys"
    echo "    chmod 600 ~/.ssh/authorized_keys"
    echo ""
    echo "  Then test:"
    echo "    ssh $VM_USER@$VM_HOST 'echo Works!'"
    echo ""
    exit 1
fi

