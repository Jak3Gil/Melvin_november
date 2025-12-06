#!/bin/bash
# setup_and_test_motors.sh - Complete motor setup and test
# Run this ON THE JETSON directly

echo "╔═══════════════════════════════════════════╗"
echo "║  Robstride Motor Setup & Test            ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# Must run as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

echo "1️⃣  Loading CH340 driver..."
modprobe ch341 2>&1 || modprobe ch340 2>&1 || echo "   (Driver may already be loaded)"

if lsmod | grep -q ch34; then
    echo "   ✅ CH340 driver loaded"
else
    echo "   ⚠️  CH340 driver not found, but continuing..."
fi

echo ""
echo "2️⃣  Checking USB serial device..."
ls -la /dev/ttyUSB* 2>&1 || echo "   ❌ No ttyUSB devices"

SERIAL_PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -1)

if [ -z "$SERIAL_PORT" ]; then
    echo "   ❌ No serial port found!"
    exit 1
fi

echo "   ✅ Using: $SERIAL_PORT"

echo ""
echo "3️⃣  Testing serial communication..."

# Create inline test program
cat > /tmp/jog_motor.c << 'CCODE'
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <serial_port> <motor_id>\n", argv[0]);
        return 1;
    }
    
    const char* port = argv[1];
    int motor_id = atoi(argv[2]);
    
    int fd = open(port, O_RDWR | O_NOCTTY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    struct termios tty;
    tcgetattr(fd, &tty);
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    tty.c_cflag = CS8 | CREAD | CLOCAL;
    tty.c_lflag = 0;
    tty.c_iflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VTIME] = 10;
    tty.c_cc[VMIN] = 0;
    tcsetattr(fd, TCSANOW, &tty);
    
    printf("Testing Motor %d on %s\n", motor_id, port);
    printf("═══════════════════════════════════════════\n\n");
    
    // Move command (from Motor Studio log)
    uint8_t move[] = {0x41, 0x54, 0x90, 0x07, 0xe8, motor_id, 0x08, 0x05,
                      0x70, 0x00, 0x00, 0x07, 0x01, 0x80, 0xa3, 0x0d, 0x0a};
    
    printf("🚀 Sending MOVE (speed=0x80A3)...\n");
    write(fd, move, sizeof(move));
    sleep(2);
    
    // Stop command
    uint8_t stop[] = {0x41, 0x54, 0x90, 0x07, 0xe8, motor_id, 0x08, 0x05,
                      0x70, 0x00, 0x00, 0x07, 0x00, 0x7f, 0xff, 0x0d, 0x0a};
    
    printf("🛑 Sending STOP...\n");
    write(fd, stop, sizeof(stop));
    
    printf("\n✅ Commands sent to motor %d\n", motor_id);
    printf("   Did you see movement?\n\n");
    
    close(fd);
    return 0;
}
CCODE

gcc -O2 -o /tmp/jog_motor /tmp/jog_motor.c

if [ ! -f /tmp/jog_motor ]; then
    echo "   ❌ Compilation failed"
    exit 1
fi

echo "   ✅ Test program ready"

echo ""
echo "4️⃣  Testing Motor 12..."
/tmp/jog_motor "$SERIAL_PORT" 12

echo ""
read -p "Did motor 12 move? (yes/no): " m12_moved

echo ""
echo "5️⃣  Testing Motor 14..."
/tmp/jog_motor "$SERIAL_PORT" 14

echo ""
read -p "Did motor 14 move? (yes/no): " m14_moved

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║  Results                                  ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "Motor 12: $m12_moved"
echo "Motor 14: $m14_moved"
echo ""

if [ "$m12_moved" = "yes" ] || [ "$m14_moved" = "yes" ]; then
    echo "✅ SUCCESS! Motors responding!"
    echo ""
    read -p "What does motor 12 control? " m12_desc
    read -p "What does motor 14 control? " m14_desc
    
    echo ""
    echo "Motor 12: $m12_desc"
    echo "Motor 14: $m14_desc"
    echo ""
    echo "Next step: Integrate with Melvin brain!"
else
    echo "⚠️  Motors still not moving"
    echo ""
    echo "Additional checks:"
    echo "  1. Are motors 12 & 14 powered? (Check LEDs)"
    echo "  2. Is Robstride adapter connected to THIS Jetson?"
    echo "  3. Try Robstride Motor Studio software to confirm"
fi

echo ""

