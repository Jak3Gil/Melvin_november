#!/bin/bash
# Try to find the VM's IP address

echo "=========================================="
echo "Finding Melvin Linux VM IP Address"
echo "=========================================="
echo ""

# Method 1: Check ARP table
echo "Method 1: Checking ARP table..."
ARP_IPS=$(arp -a | grep "192.168.64" | awk '{print $2}' | tr -d '()')
if [ -n "$ARP_IPS" ]; then
    echo "Found potential VM IPs:"
    for ip in $ARP_IPS; do
        echo "  - $ip"
    done
else
    echo "  No IPs found in ARP table yet"
fi
echo ""

# Method 2: Try common UTM IP ranges
echo "Method 2: Common UTM VM IPs (try these):"
echo "  - 192.168.64.2"
echo "  - 192.168.64.3"
echo "  - 192.168.64.4"
echo ""

# Method 3: Try to ping and test SSH
echo "Method 3: Testing connections..."
for ip in 192.168.64.{2..10}; do
    if ping -c 1 -W 1 "$ip" &>/dev/null; then
        echo "  ✅ $ip is responding to ping"
        # Try SSH
        if timeout 2 ssh -o ConnectTimeout=1 -o StrictHostKeyChecking=no "ubuntu@$ip" "echo 'SSH works!'" &>/dev/null 2>&1; then
            echo "  ✅✅ $ip - SSH CONNECTED!"
            echo ""
            echo "=========================================="
            echo "VM IP FOUND: $ip"
            echo "=========================================="
            echo ""
            echo "SSH into VM:"
            echo "  ssh ubuntu@$ip"
            echo ""
            echo "Transfer Melvin:"
            echo "  ./transfer_and_test.sh $ip"
            echo ""
            exit 0
        fi
    fi
done

echo ""
echo "=========================================="
echo "Could not auto-detect VM IP"
echo "=========================================="
echo ""
echo "To find it manually:"
echo "  1. Look at the VM window in UTM"
echo "  2. Inside the VM terminal, run:"
echo "     ip addr show | grep 'inet ' | grep -v '127.0.0.1'"
echo ""
echo "Or check UTM's network settings for the VM IP"
echo ""




