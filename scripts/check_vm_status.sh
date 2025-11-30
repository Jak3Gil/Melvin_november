#!/bin/bash
# Check VM status and connection info

echo "=========================================="
echo "Melvin Linux VM Status"
echo "=========================================="
echo ""

# Check if VM is running
QEMU_PID=$(pgrep -f "qemu-aarch64.*melvinlinux" | head -1)

if [ -z "$QEMU_PID" ]; then
    echo "❌ VM is not running"
    echo "   Start it from UTM first"
    exit 1
fi

echo "✅ VM is RUNNING"
echo "   Process ID: $QEMU_PID"
echo ""

# Get VM resource usage
VM_STATS=$(ps -p $QEMU_PID -o %cpu,%mem,command 2>/dev/null | tail -1)
CPU=$(echo $VM_STATS | awk '{print $1}')
MEM=$(echo $VM_STATS | awk '{print $2}')

echo "📊 Resource Usage:"
echo "   CPU: ${CPU}%"
echo "   Memory: ${MEM}%"
echo ""

# Try to find VM IP from network interfaces
echo "🔍 Checking network interfaces..."
echo ""

# Check vmnet interfaces
VMNET_IP=$(ifconfig | grep -A 3 "vmnet" | grep "inet " | awk '{print $2}' | head -1)

if [ -n "$VMNET_IP" ]; then
    echo "   vmnet interface found"
    # VM IPs typically start with 192.168.64.x or 10.0.2.x
    echo ""
    echo "📍 Common VM IP ranges:"
    echo "   - 192.168.64.x (UTM default)"
    echo "   - 10.0.2.x (QEMU NAT)"
    echo ""
    echo "   To find exact IP, check inside the VM:"
    echo "   ip addr show | grep 'inet '"
fi

echo ""
echo "=========================================="
echo "Quick Commands:"
echo "=========================================="
echo ""
echo "To find VM IP (inside VM):"
echo "  ip addr show | grep 'inet ' | grep -v '127.0.0.1'"
echo ""
echo "To SSH into VM (once you have IP):"
echo "  ssh ubuntu@<vm_ip>"
echo ""
echo "To transfer Melvin:"
echo "  ./transfer_and_test.sh <vm_ip>"
echo ""
echo "=========================================="




