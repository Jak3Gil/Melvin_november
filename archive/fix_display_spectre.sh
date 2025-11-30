#!/bin/bash
# Fix display for Spectre 24" monitor - test both HDMI ports

set -e

echo "=========================================="
echo "Spectre 24\" Display Fix - Systematic Test"
echo "=========================================="
echo ""

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "Current Status:"
echo "  - Framebuffer exists and can be written to"
echo "  - But nothing appears on your Spectre display"
echo "  - This suggests wrong HDMI port or resolution issue"
echo ""

echo "Step 1: Testing HDMI-A-1 with 1280x720@60..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    # Set to HDMI-A-1, simple resolution
    sudo sed -i 's/video=HDMI-A-[12]:[^ ]*/video=HDMI-A-1:1280x720@60/' /boot/extlinux/extlinux.conf
    sudo sed -i 's/video=DP-0:[^ ]*/video=HDMI-A-1:1280x720@60/' /boot/extlinux/extlinux.conf
    
    echo "✓ Set to HDMI-A-1:1280x720@60"
    echo ""
    echo "Current boot config:"
    sudo grep -E 'video=' /boot/extlinux/extlinux.conf | head -1
EOF
echo ""

echo "Step 2: Create test script for after reboot..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    cat > /home/melvin/test_display_after_reboot.sh << 'TESTEOF'
#!/bin/bash
echo "=== Display Test After Reboot ==="
echo ""

# Fill screen with test pattern
cat > /tmp/display_test.c << 'CEOF'
#include <fcntl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main() {
    int fd = open("/dev/fb0", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    
    struct fb_fix_screeninfo finfo;
    struct fb_var_screeninfo vinfo;
    ioctl(fd, FBIOGET_FSCREENINFO, &finfo);
    ioctl(fd, FBIOGET_VSCREENINFO, &vinfo);
    
    char *fb = mmap(0, finfo.smem_len, PROT_WRITE, MAP_SHARED, fd, 0);
    if (fb == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    
    // Fill with blue
    memset(fb, 0, finfo.smem_len);
    for (int i = 0; i < finfo.smem_len; i += 4) {
        fb[i] = 0xFF;     // B
        fb[i+1] = 0x00;   // G
        fb[i+2] = 0x00;   // R
        fb[i+3] = 0xFF;   // A
    }
    
    printf("Screen filled with BLUE - check your display!\n");
    printf("Resolution: %dx%d\n", vinfo.xres, vinfo.yres);
    sleep(5);
    
    // Fill with green
    for (int i = 0; i < finfo.smem_len; i += 4) {
        fb[i] = 0x00;     // B
        fb[i+1] = 0xFF;   // G
        fb[i+2] = 0x00;   // R
        fb[i+3] = 0xFF;   // A
    }
    printf("Screen filled with GREEN\n");
    sleep(5);
    
    // Fill with red
    for (int i = 0; i < finfo.smem_len; i += 4) {
        fb[i] = 0x00;     // B
        fb[i+1] = 0x00;   // G
        fb[i+2] = 0xFF;   // R
        fb[i+3] = 0xFF;   // A
    }
    printf("Screen filled with RED\n");
    sleep(5);
    
    // Clear to black
    memset(fb, 0, finfo.smem_len);
    printf("Screen cleared to BLACK\n");
    
    munmap(fb, finfo.smem_len);
    close(fd);
    return 0;
}
CEOF
    
    gcc -o /tmp/display_test /tmp/display_test.c
    sudo /tmp/display_test
    
    echo ""
    echo "Did you see blue, green, red screens on your display?"
    echo "If YES: Display is working! If NO: Try other HDMI port"
TESTEOF
    
    chmod +x /home/melvin/test_display_after_reboot.sh
    echo "✓ Test script created at ~/test_display_after_reboot.sh"
EOF
echo ""

echo "Step 3: Instructions..."
echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo ""
echo "1. REBOOT the Jetson:"
echo "   sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'sudo reboot'"
echo ""
echo "2. After reboot, check your Spectre display"
echo "   - You should see console output or colored screens"
echo ""
echo "3. If still nothing, try the OTHER HDMI port:"
echo "   - Physically move the HDMI cable to the other HDMI port"
echo "   - Then run this script again with option 2"
echo ""
echo "4. Test display after reboot:"
echo "   sshpass -p '$JETSON_PASS' ssh $JETSON_HOST '~/test_display_after_reboot.sh'"
echo ""
echo "=========================================="
echo "Troubleshooting Checklist:"
echo "=========================================="
echo "□ HDMI cable is firmly connected"
echo "□ Display is powered on"
echo "□ Display input is set to correct HDMI port"
echo "□ Try BOTH HDMI ports on Jetson"
echo "□ Try different HDMI cable (if available)"
echo "□ Check if display works with another device"
echo ""

