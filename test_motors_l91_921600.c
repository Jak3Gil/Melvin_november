/*
 * test_motors_l91_921600.c - Control Motors 12 and 14 using L91 protocol
 * 
 * Based on Motor Studio logs - uses AT commands over serial at 921600 baud
 * Format: AT <cmd> <addr> <can_id> <len> <data> \r\n
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
#include <math.h>

#define SERIAL_PORT "/dev/ttyUSB0"

// Baud rate - use 921600 if available, fallback to 115200
#ifdef B921600
#define BAUD_RATE B921600
#else
// For systems without B921600, we'll set it manually
#define BAUD_RATE B115200
#define USE_CUSTOM_BAUD 921600
#endif

// Motor CAN IDs (from Motor Studio log)
#define MOTOR_5_CAN_ID   0x05  // Motor 5 = CAN ID 5 (0x05)
#define MOTOR_12_CAN_ID  0x0C  // Motor 12 = CAN ID 12 (0x0C)
#define MOTOR_13_CAN_ID  0x0D  // Motor 13 = CAN ID 13 (0x0D)
#define MOTOR_14_CAN_ID  0x0E  // Motor 14 = CAN ID 14 (0x0E)

static int serial_fd = -1;
static bool running = true;

static void signal_handler(int sig) {
    printf("\n🛑 Stopping motors...\n");
    running = false;
}

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
    
    // Set baud rate to 921600
#ifdef USE_CUSTOM_BAUD
    // For systems without B921600 constant, set it manually
    speed_t baud = USE_CUSTOM_BAUD;
    if (cfsetspeed(&tty, baud) != 0) {
        // Fallback: try to set using cfsetospeed/cfsetispeed with closest value
        cfsetospeed(&tty, B115200);
        cfsetispeed(&tty, B115200);
        printf("⚠️  Using 115200 baud (921600 not available on this system)\n");
    }
#else
    cfsetospeed(&tty, BAUD_RATE);
    cfsetispeed(&tty, BAUD_RATE);
#endif
    
    // 8N1 mode
    tty.c_cflag &= ~PARENB;        // No parity
    tty.c_cflag &= ~CSTOPB;        // 1 stop bit
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;            // 8 data bits
    tty.c_cflag &= ~CRTSCTS;       // No hardware flow control
    tty.c_cflag |= CREAD | CLOCAL; // Enable reading
    
    // Raw mode
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG;
    
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);
    
    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;
    
    // Read timeouts
    tty.c_cc[VTIME] = 10;    // 1 second timeout
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
        perror("Serial write");
        return false;
    }
    
    tcdrain(serial_fd);
    usleep(10000);  // 10ms delay
    
    return true;
}

/* Send MOVE_JOG command (from Motor Studio log) */
/* Format: AT 90 07 e8 <can_id> 08 05 70 00 00 07 <flag> <speed_bytes> 0d 0a */
static bool move_motor_jog(uint8_t can_id, float speed, uint8_t flag) {
    uint8_t cmd[20];
    int idx = 0;
    
    cmd[idx++] = 0x41;  // 'A'
    cmd[idx++] = 0x54;  // 'T'
    cmd[idx++] = 0x90;  // Command type (MOVE_JOG)
    cmd[idx++] = 0x07;  // Address high
    cmd[idx++] = 0xe8;  // Address low (0x07e8)
    cmd[idx++] = can_id;  // Motor CAN ID
    cmd[idx++] = 0x08;  // Data length
    cmd[idx++] = 0x05;  // Command high
    cmd[idx++] = 0x70;  // Command low (MOVE_JOG = 0x0570)
    cmd[idx++] = 0x00;  // Fixed
    cmd[idx++] = 0x00;  // Fixed
    cmd[idx++] = 0x07;  // Fixed
    cmd[idx++] = flag;  // 0=stop, 1=move
    
    // Speed encoding (from log: 0.1 = 0x80a3, -0.1 = 0x7f5b)
    // Speed appears to be 16-bit signed, where 0x7fff = 0.0
    int16_t speed_val;
    if (speed == 0.0f) {
        speed_val = 0x7fff;  // Stop
    } else if (speed > 0.0f) {
        // Positive speed: 0x80a3 = 0.1, so scale accordingly
        speed_val = 0x8000 + (int16_t)(speed * 3283.0f);  // Approximate scaling
    } else {
        // Negative speed: 0x7f5b = -0.1
        speed_val = 0x7fff + (int16_t)(speed * 3283.0f);
    }
    
    cmd[idx++] = (speed_val >> 8) & 0xFF;  // High byte
    cmd[idx++] = speed_val & 0xFF;         // Low byte
    cmd[idx++] = 0x0d;  // \r
    cmd[idx++] = 0x0a;  // \n
    
    return send_l91_command(cmd, idx);
}

/* Activate motor (from Motor Studio log) */
/* Format: AT 00 07 e8 <can_id> 01 00 0d 0a */
static bool activate_motor(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x00, 0x07, 0xe8, can_id, 0x01, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

/* Load parameters (from Motor Studio log) */
/* Format: AT 20 07 e8 <can_id> 08 00 c4 00 00 00 00 00 00 0d 0a */
static bool load_params(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x20, 0x07, 0xe8, can_id, 0x08, 0x00,
                     0xc4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

/* Test motor with slow movements */
static void test_motor(uint8_t can_id, const char* name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing %s (CAN ID 0x%02X)\n", name, can_id);
    printf("═══════════════════════════════════════════\n\n");
    
    // Activate motor
    printf("🔧 Activating motor... ");
    fflush(stdout);
    if (activate_motor(can_id)) {
        printf("✓\n");
    } else {
        printf("✗\n");
        return;
    }
    
    usleep(200000);  // 200ms
    
    // Load parameters
    printf("📋 Loading parameters... ");
    fflush(stdout);
    if (load_params(can_id)) {
        printf("✓\n");
    } else {
        printf("✗\n");
    }
    
    usleep(200000);  // 200ms
    
    printf("\n🔹 Small forward movement (speed = 0.05)\n\n");
    
    for (int i = 0; i < 2 && running; i++) {
        printf("   Move %d: ", i + 1);
        fflush(stdout);
        
        if (move_motor_jog(can_id, 0.05f, 1)) {
            printf("✓ Moving forward (small)\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);  // 500ms - shorter movement
        
        // Stop
        move_motor_jog(can_id, 0.0f, 0);
        printf("   Stop: ✓\n");
        usleep(500000);
    }
    
    if (!running) return;
    
    printf("\n🔹 Small backward movement (speed = -0.05)\n\n");
    
    for (int i = 0; i < 2 && running; i++) {
        printf("   Move %d: ", i + 1);
        fflush(stdout);
        
        if (move_motor_jog(can_id, -0.05f, 1)) {
            printf("✓ Moving backward (small)\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);  // 500ms - shorter movement
        
        // Stop
        move_motor_jog(can_id, 0.0f, 0);
        printf("   Stop: ✓\n");
        usleep(500000);
    }
    
    printf("\n✅ Test complete for %s\n\n", name);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  L91 Protocol Motor Test (921600 baud)    ║\n");
    printf("║  Motors 5 and 13 (small movements)      ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_serial()) {
        fprintf(stderr, "❌ Failed to initialize serial port\n");
        return 1;
    }
    
    printf("⚠️  Ready to test motors\n");
    printf("   Watch the robot and note what moves!\n");
    printf("\n");
    printf("Press Enter to test Motor 5...\n");
    getchar();
    
    if (running) {
        test_motor(MOTOR_5_CAN_ID, "Motor 5");
    }
    
    if (!running) goto cleanup;
    
    printf("Press Enter to test Motor 13...\n");
    getchar();
    
    if (running) {
        test_motor(MOTOR_13_CAN_ID, "Motor 13");
    }
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Test Complete!                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("📝 What did you observe?\n");
    printf("   Motor 5 controls: __________________\n");
    printf("   Motor 13 controls: __________________\n");
    printf("\n");
    
cleanup:
    if (serial_fd >= 0) {
        printf("🔒 Stopping all motors...\n");
        move_motor_jog(MOTOR_5_CAN_ID, 0.0f, 0);
        move_motor_jog(MOTOR_13_CAN_ID, 0.0f, 0);
        usleep(100000);
        close(serial_fd);
    }
    
    printf("✅ Safe shutdown complete\n");
    return 0;
}

