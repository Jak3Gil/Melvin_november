#!/bin/bash
# Diagnostic script to test Jetson connection methods
# Tests all available connection methods and provides troubleshooting steps

set -e

echo "=========================================="
echo "Jetson Connection Diagnostic Tool"
echo "=========================================="
echo ""

# Configuration
JETSON_IP_ETHERNET="169.254.123.100"
JETSON_IP_USB="192.168.55.1"
JETSON_USER="melvin"
JETSON_PASS="123456"

# Track connection status
ETHERNET_OK=false
USB_NETWORK_OK=false
SERIAL_AVAILABLE=false
SSH_WORKING=false

echo "Testing connection methods..."
echo ""

# Test 1: Ethernet connection (169.254.123.100)
echo "1. Testing Ethernet connection ($JETSON_IP_ETHERNET)..."
if ping -c 2 -W 2 "$JETSON_IP_ETHERNET" &>/dev/null; then
    echo "   ✓ Ethernet ping successful"
    ETHERNET_OK=true
    
    # Test SSH
    echo "   Testing SSH connection..."
    if sshpass -p "$JETSON_PASS" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP_ETHERNET" "echo 'SSH OK'" &>/dev/null 2>&1; then
        echo "   ✓ SSH connection working"
        SSH_WORKING=true
    else
        echo "   ✗ SSH connection failed"
    fi
else
    echo "   ✗ Ethernet ping failed"
fi
echo ""

# Test 2: USB network connection (192.168.55.1)
echo "2. Testing USB network connection ($JETSON_IP_USB)..."
if ping -c 2 -W 2 "$JETSON_IP_USB" &>/dev/null; then
    echo "   ✓ USB network ping successful"
    USB_NETWORK_OK=true
    
    # Test SSH
    if [ "$SSH_WORKING" = false ]; then
        echo "   Testing SSH connection..."
        if sshpass -p "$JETSON_PASS" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP_USB" "echo 'SSH OK'" &>/dev/null 2>&1; then
            echo "   ✓ SSH connection working"
            SSH_WORKING=true
        else
            echo "   ✗ SSH connection failed"
        fi
    fi
else
    echo "   ✗ USB network ping failed"
fi
echo ""

# Test 3: Serial/USB devices (COM8 on Windows, /dev/cu.* on Mac)
echo "3. Checking for serial/USB devices..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SERIAL_DEVICES=$(ls /dev/cu.usb* /dev/tty.usb* 2>/dev/null || true)
    if [ -n "$SERIAL_DEVICES" ]; then
        echo "   ✓ Found serial devices:"
        for dev in $SERIAL_DEVICES; do
            echo "     - $dev"
        done
        SERIAL_AVAILABLE=true
    else
        echo "   ✗ No serial devices found"
        echo "     (Check if Jetson is connected via USB)"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    SERIAL_DEVICES=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true)
    if [ -n "$SERIAL_DEVICES" ]; then
        echo "   ✓ Found serial devices:"
        for dev in $SERIAL_DEVICES; do
            echo "     - $dev"
        done
        SERIAL_AVAILABLE=true
    else
        echo "   ✗ No serial devices found"
    fi
else
    # Windows (COM8 mentioned in memories)
    echo "   Note: On Windows, check for COM8 or other COM ports"
    echo "   Use Device Manager to find the port"
fi
echo ""

# Summary and recommendations
echo "=========================================="
echo "Diagnostic Summary"
echo "=========================================="
echo ""

if [ "$SSH_WORKING" = true ]; then
    echo "✓ WORKING: SSH connection is available"
    if [ "$ETHERNET_OK" = true ]; then
        echo "  Use: ssh $JETSON_USER@$JETSON_IP_ETHERNET"
    elif [ "$USB_NETWORK_OK" = true ]; then
        echo "  Use: ssh $JETSON_USER@$JETSON_IP_USB"
    fi
    echo ""
    echo "You can now:"
    echo "  - Deploy files: ./deploy_jetson.sh"
    echo "  - Run commands: sshpass -p '$JETSON_PASS' ssh $JETSON_USER@$JETSON_IP_ETHERNET 'command'"
    exit 0
fi

echo "✗ PROBLEM: No working SSH connection detected"
echo ""

# Troubleshooting steps
echo "Troubleshooting Steps:"
echo ""

if [ "$ETHERNET_OK" = false ] && [ "$USB_NETWORK_OK" = false ]; then
    echo "1. Network Connection Issues:"
    echo "   - Check if Jetson is powered on"
    echo "   - Verify ethernet cable is connected (for $JETSON_IP_ETHERNET)"
    echo "   - Verify USB cable is connected (for $JETSON_IP_USB)"
    echo "   - Check if Jetson has booted completely"
    echo ""
    
    echo "2. Check Jetson Status:"
    if [ "$SERIAL_AVAILABLE" = true ]; then
        echo "   ✓ Serial device found - you can use serial console"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "   Connect with: screen $(ls /dev/cu.usb* | head -1) 115200"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "   Connect with: screen $(ls /dev/ttyUSB* /dev/ttyACM* | head -1) 115200"
        fi
        echo ""
        echo "   Once connected via serial:"
        echo "   - Check if Jetson is booted: uname -a"
        echo "   - Check IP address: ip addr show"
        echo "   - Check SSH service: sudo systemctl status ssh"
        echo "   - Start SSH if needed: sudo systemctl start ssh"
    else
        echo "   ✗ No serial device found"
        echo "   - Connect Jetson via USB to your computer"
        echo "   - On Mac: Check /dev/cu.usb* devices"
        echo "   - On Windows: Check Device Manager for COM ports"
    fi
    echo ""
fi

if [ "$ETHERNET_OK" = true ] || [ "$USB_NETWORK_OK" = true ]; then
    echo "3. SSH Service Issues:"
    echo "   Jetson is reachable but SSH is not working"
    echo ""
    echo "   Try connecting via serial console first:"
    if [ "$SERIAL_AVAILABLE" = true ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "   screen $(ls /dev/cu.usb* | head -1) 115200"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "   screen $(ls /dev/ttyUSB* /dev/ttyACM* | head -1) 115200"
        fi
    else
        echo "   (Connect USB cable first to get serial access)"
    fi
    echo ""
    echo "   Once connected via serial, run on Jetson:"
    echo "   sudo systemctl start ssh"
    echo "   sudo systemctl enable ssh"
    echo "   sudo systemctl status ssh"
    echo ""
fi

echo "4. Alternative: Direct USB Connection"
echo "   If Jetson is in recovery mode or not booted:"
echo "   - Put Jetson in Recovery Mode (hold RECOVERY button while powering on)"
echo "   - Use SDK Manager to flash/reinstall JetPack"
echo "   - See: SETUP_JETSON_VM.md for VM setup instructions"
echo ""

echo "5. Check Network Configuration:"
echo "   If Jetson IP has changed, you may need to:"
echo "   - Connect via serial to check current IP: ip addr show"
echo "   - Or scan network: nmap -sn 169.254.0.0/16"
echo ""

echo "=========================================="
echo "Quick Connection Commands"
echo "=========================================="
echo ""
echo "If SSH works, use:"
echo "  sshpass -p '$JETSON_PASS' ssh $JETSON_USER@$JETSON_IP_ETHERNET"
echo ""
echo "If serial works, use:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -n "$SERIAL_DEVICES" ]; then
        echo "  screen $(echo $SERIAL_DEVICES | awk '{print $1}') 115200"
    else
        echo "  screen /dev/cu.usb* 115200  (find the correct device)"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -n "$SERIAL_DEVICES" ]; then
        echo "  screen $(echo $SERIAL_DEVICES | awk '{print $1}') 115200"
    else
        echo "  screen /dev/ttyUSB0 115200  (or /dev/ttyACM0)"
    fi
else
    echo "  (Use PuTTY or similar on Windows with COM8, 115200 baud)"
fi
echo ""

