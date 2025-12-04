/*
 * test_robstride_l91.c - Robstride L91 Protocol via Serial
 * 
 * Based on Motor Studio logs - uses AT commands over serial
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

// L91 Commands (from your working log)
#define L91_LOAD_PARAM  0xC4  // Load parameters
#define L91_MOVE_JOG    0x0570  // Jog movement (05 70)

static int serial_fd = -1;
static bool running = true;

static void signal_handler(int sig) {
    printf("\n🛑 Stopping...\n");
    running = false;
}

/* Initialize serial port */
static bool init_serial(void) {
    struct termios tty;
    
    printf("📡 Opening serial port: %s\n", SERIAL_PORT);
    
    serial_fd = open(SERIAL_PORT, O_RDWR | O_NOCTTY);
    if (serial_fd < 0) {
        perror("Serial port");
        return false;
    }
    
    // Get current settings
    if (tcgetattr(serial_fd, &tty) != 0) {
        perror("tcgetattr");
        close(serial_fd);
        return false;
    }
    
    // Set baud rate (commonly 115200 for Robstride)
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    
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
    
    printf("✅ Serial port ready (115200 baud)\n\n");
    return true;
}

/* Send AT command */
static bool send_at_command(const uint8_t* cmd, size_t len) {
    // Flush input
    tcflush(serial_fd, TCIFLUSH);
    
    // Send command
    ssize_t written = write(serial_fd, cmd, len);
    if (written != len) {
        perror("Serial write");
        return false;
    }
    
    // Wait for command to be sent
    tcdrain(serial_fd);
    usleep(10000);  // 10ms
    
    return true;
}

/* Read response */
static int read_response(uint8_t* buffer, size_t max_len, int timeout_ms) {
    fd_set readfds;
    struct timeval timeout;
    
    FD_ZERO(&readfds);
    FD_SET(serial_fd, &readfds);
    
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    
    if (select(serial_fd + 1, &readfds, NULL, NULL, &timeout) > 0) {
        return read(serial_fd, buffer, max_len);
    }
    
    return 0;
}

/* Detect motors (L91 protocol) */
static bool detect_motors(void) {
    printf("🔍 Detecting motors...\n\n");
    
    // AT command to detect devices (from your log)
    // AT+AT\r\n
    uint8_t detect_cmd[] = {0x41, 0x54, 0x2b, 0x41, 0x54, 0x0d, 0x0a};
    
    printf("Sending: AT+AT\\r\\n\n");
    
    if (!send_at_command(detect_cmd, sizeof(detect_cmd))) {
        return false;
    }
    
    // Wait for response
    uint8_t response[256];
    int len = read_response(response, sizeof(response), 2000);
    
    if (len > 0) {
        printf("✅ Response (%d bytes):\n   ", len);
        for (int i = 0; i < len; i++) {
            printf("%02X ", response[i]);
            if ((i + 1) % 16 == 0) printf("\n   ");
        }
        printf("\n\n");
        return true;
    } else {
        printf("❌ No response\n\n");
        return false;
    }
}

/* Move motor using L91 MOVE_JOG */
static bool move_motor_jog(uint8_t can_id, float speed, uint8_t flag) {
    // Build L91 MOVE_JOG command (from your log)
    // AT <90> <07 e8> <0c> <08> <05 70> <data> <0d 0a>
    uint8_t cmd[20];
    int idx = 0;
    
    cmd[idx++] = 0x41;  // 'A'
    cmd[idx++] = 0x54;  // 'T'
    cmd[idx++] = 0x90;  // Flags
    cmd[idx++] = 0x07;  // ID high
    cmd[idx++] = 0xe8;  // ID low (0x07e8 = 2024)
    cmd[idx++] = can_id;  // Motor CAN ID
    cmd[idx++] = 0x08;  // Length
    cmd[idx++] = 0x05;  // Command high
    cmd[idx++] = 0x70;  // Command low (MOVE_JOG = 0x0570)
    
    // Data (from log): 00 00 07 <flag> <speed_bytes>
    cmd[idx++] = 0x00;
    cmd[idx++] = 0x00;
    cmd[idx++] = 0x07;
    cmd[idx++] = flag;  // 0=stop, 1=move
    
    // Speed (16-bit, little endian from log)
    uint16_t speed_val = (uint16_t)(speed * 655.35f);  // Scale 0-1 to 0-65535
    cmd[idx++] = speed_val >> 8;
    cmd[idx++] = speed_val & 0xFF;
    
    cmd[idx++] = 0x0d;  // \r
    cmd[idx++] = 0x0a;  // \n
    
    return send_at_command(cmd, idx);
}

/* Test motor */
static void test_motor(uint8_t can_id, const char* name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing %s (CAN ID %d)\n", name, can_id);
    printf("═══════════════════════════════════════════\n\n");
    
    printf("🔹 Slow forward movement\n\n");
    
    for (int i = 0; i < 5 && running; i++) {
        printf("   Step %d: Moving... ", i + 1);
        fflush(stdout);
        
        if (move_motor_jog(can_id, 0.1f, 1)) {  // speed=0.1, flag=1 (move)
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        sleep(1);
        
        // Stop
        move_motor_jog(can_id, 0.0f, 0);  // speed=0, flag=0 (stop)
        sleep(1);
    }
    
    printf("\n✅ Test complete\n\n");
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Robstride L91 Protocol Test             ║\n");
    printf("║  Using Serial AT Commands                ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_serial()) {
        return 1;
    }
    
    // Detect motors first
    if (!detect_motors()) {
        printf("⚠️  Detection failed, but continuing...\n\n");
    }
    
    sleep(1);
    
    printf("Press Enter to test Motor 12...\n");
    getchar();
    
    if (running) test_motor(12, "Motor 12");
    if (!running) goto cleanup;
    
    printf("Press Enter to test Motor 14...\n");
    getchar();
    
    if (running) test_motor(14, "Motor 14");
    
    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║         Did you see movement?             ║\n");
    printf("╚═══════════════════════════════════════════╝\n\n");
    
cleanup:
    if (serial_fd >= 0) {
        // Stop all motors
        move_motor_jog(12, 0.0f, 0);
        move_motor_jog(14, 0.0f, 0);
        close(serial_fd);
    }
    
    return 0;
}

