/*
 * detect_all_motors_l91.c - Detect all motors using L91 protocol
 * 
 * Scans all possible CAN IDs to find which motors are actually connected
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <signal.h>

#define SERIAL_PORT "/dev/ttyUSB0"

#ifdef B921600
#define BAUD_RATE B921600
#else
#define BAUD_RATE B115200
#define USE_CUSTOM_BAUD 921600
#endif

static int serial_fd = -1;

/* Initialize serial port at 921600 baud */
static bool init_serial(void) {
    struct termios tty;
    
    printf("📡 Opening serial port: %s at 921600 baud\n", SERIAL_PORT);
    
    serial_fd = open(SERIAL_PORT, O_RDWR | O_NOCTTY);
    if (serial_fd < 0) {
        perror("Serial port");
        return false;
    }
    
    if (tcgetattr(serial_fd, &tty) != 0) {
        perror("tcgetattr");
        close(serial_fd);
        return false;
    }
    
#ifdef USE_CUSTOM_BAUD
    speed_t baud = USE_CUSTOM_BAUD;
    if (cfsetspeed(&tty, baud) != 0) {
        cfsetospeed(&tty, B115200);
        cfsetispeed(&tty, B115200);
    }
#else
    cfsetospeed(&tty, BAUD_RATE);
    cfsetispeed(&tty, BAUD_RATE);
#endif
    
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;
    
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG;
    
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);
    
    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;
    
    tty.c_cc[VTIME] = 10;
    tty.c_cc[VMIN] = 0;
    
    if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        close(serial_fd);
        return false;
    }
    
    printf("✅ Serial port ready (921600 baud)\n\n");
    return true;
}

/* Send L91 AT command */
static bool send_l91_command(const uint8_t* cmd, size_t len) {
    tcflush(serial_fd, TCIFLUSH);
    
    ssize_t written = write(serial_fd, cmd, len);
    if (written != len) {
        return false;
    }
    
    tcdrain(serial_fd);
    usleep(50000);  // 50ms delay
    
    return true;
}

/* Detect devices using AT+AT command (from Motor Studio) */
static bool detect_devices(void) {
    printf("🔍 Sending AT+AT detection command...\n");
    
    // AT+AT\r\n
    uint8_t detect_cmd[] = {0x41, 0x54, 0x2b, 0x41, 0x54, 0x0d, 0x0a};
    
    if (!send_l91_command(detect_cmd, sizeof(detect_cmd))) {
        printf("❌ Failed to send detection command\n");
        return false;
    }
    
    printf("   Sent. Waiting for responses...\n");
    usleep(1000000);  // 1 second
    
    // Try to read response
    uint8_t response[512];
    int got = read(serial_fd, response, sizeof(response));
    
    if (got > 0) {
        printf("   ✅ Response (%d bytes):\n      ", got);
        for (int i = 0; i < got && i < 100; i++) {
            printf("%02X ", response[i]);
            if ((i + 1) % 16 == 0) printf("\n      ");
        }
        printf("\n\n");
        return true;
    } else {
        printf("   ⚠️  No immediate response (this is normal)\n\n");
        return true;  // Continue anyway
    }
}

/* Activate motor and check if it responds */
static bool test_motor_id(uint8_t can_id) {
    // Activate command: AT 00 07 e8 <can_id> 01 00 0d 0a
    uint8_t activate_cmd[] = {0x41, 0x54, 0x00, 0x07, 0xe8, can_id, 0x01, 0x00, 0x0d, 0x0a};
    
    if (!send_l91_command(activate_cmd, sizeof(activate_cmd))) {
        return false;
    }
    
    usleep(200000);  // 200ms
    
    // Try to read any response
    uint8_t response[256];
    int got = read(serial_fd, response, sizeof(response));
    
    // If we got a response, motor might be there
    // Also try a small move command to see if motor responds
    uint8_t move_cmd[] = {0x41, 0x54, 0x90, 0x07, 0xe8, can_id, 0x08, 0x05,
                          0x70, 0x00, 0x00, 0x07, 0x01, 0x80, 0xa3, 0x0d, 0x0a};
    send_l91_command(move_cmd, sizeof(move_cmd));
    
    usleep(300000);  // 300ms
    
    // Stop immediately
    uint8_t stop_cmd[] = {0x41, 0x54, 0x90, 0x07, 0xe8, can_id, 0x08, 0x05,
                          0x70, 0x00, 0x00, 0x07, 0x00, 0x7f, 0xff, 0x0d, 0x0a};
    send_l91_command(stop_cmd, sizeof(stop_cmd));
    
    return (got > 0);  // Return true if we got any response
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  L91 Motor Detection Scan                  ║\n");
    printf("║  Finding all connected motors             ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    if (!init_serial()) {
        fprintf(stderr, "❌ Failed to initialize serial port\n");
        return 1;
    }
    
    // Step 1: Try AT+AT detection
    detect_devices();
    
    printf("═══════════════════════════════════════════\n");
    printf("Scanning CAN IDs 1-14...\n");
    printf("(Watch for motor movement - that indicates a real motor)\n");
    printf("═══════════════════════════════════════════\n\n");
    
    printf("Press Enter to start scan...\n");
    getchar();
    
    int found_count = 0;
    
    // Scan CAN IDs 1-14 (0x01 to 0x0E)
    for (uint8_t can_id = 0x01; can_id <= 0x0E; can_id++) {
        printf("Testing CAN ID 0x%02X (Motor %d): ", can_id, can_id);
        fflush(stdout);
        
        if (test_motor_id(can_id)) {
            printf("✅ RESPONDED - Motor detected!\n");
            found_count++;
        } else {
            printf("❌ No response\n");
        }
        
        usleep(300000);  // 300ms between tests
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════\n");
    printf("📊 Scan Results:\n");
    printf("═══════════════════════════════════════════\n");
    printf("Found %d motor(s) responding\n", found_count);
    printf("\n");
    printf("💡 Which motors did you see move?\n");
    printf("   (This tells us the actual CAN IDs)\n");
    printf("\n");
    
    close(serial_fd);
    return 0;
}

