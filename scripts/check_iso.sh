#!/bin/bash
# Quick check for Ubuntu ISO files

echo "=========================================="
echo "Checking for Ubuntu ISO files..."
echo "=========================================="
echo ""

VM_DIR="$HOME/melvin_linux_vm"
ISO_2404="$VM_DIR/ubuntu-24.04.3-live-server-arm64.iso"
ISO_2204="$VM_DIR/ubuntu-22.04-server-arm64.iso"

# Check for 24.04
if [ -f "$ISO_2404" ]; then
    SIZE=$(du -h "$ISO_2404" | cut -f1)
    echo "✅ Found: ubuntu-24.04.3-live-server-arm64.iso ($SIZE)"
    
    # Verify it's a real ISO
    if file "$ISO_2404" 2>/dev/null | grep -qi "iso\|archive\|9660"; then
        echo "   ✓ Valid ISO file"
    elif [ $(stat -f%z "$ISO_2404" 2>/dev/null || stat -c%s "$ISO_2404" 2>/dev/null) -gt 1000000 ]; then
        echo "   ✓ File size looks good (>1MB)"
    else
        echo "   ⚠️  May be corrupted or incomplete"
    fi
else
    echo "❌ Not found: ubuntu-24.04.3-live-server-arm64.iso"
fi

# Check for 22.04
if [ -f "$ISO_2204" ]; then
    SIZE=$(du -h "$ISO_2204" | cut -f1)
    echo "✅ Found: ubuntu-22.04-server-arm64.iso ($SIZE)"
else
    echo "   (22.04 not found - that's OK)"
fi

echo ""

# Check Downloads folder
if [ -d "$HOME/Downloads" ]; then
    DOWNLOADS=$(find "$HOME/Downloads" -maxdepth 1 -iname "*ubuntu*24.04*.iso" 2>/dev/null)
    if [ -n "$DOWNLOADS" ]; then
        echo "📥 Found Ubuntu 24.04 ISO in Downloads:"
        echo "$DOWNLOADS" | while read iso; do
            SIZE=$(du -h "$iso" | cut -f1)
            echo "   $iso ($SIZE)"
            echo ""
            echo "   To move it to the VM directory:"
            echo "   mv \"$iso\" \"$VM_DIR/ubuntu-24.04.3-live-server-arm64.iso\""
        done
    fi
fi

echo "=========================================="
echo "VM Directory: $VM_DIR"
echo "=========================================="

if [ ! -f "$ISO_2404" ] && [ ! -f "$ISO_2204" ]; then
    echo ""
    echo "No ISO found. Download one from:"
    echo "   https://ubuntu.com/download/server/arm"
    echo ""
    echo "Or run: ./setup_linux_vm.sh"
fi

