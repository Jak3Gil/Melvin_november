/*
 * diagnose_motor_responses.c - Diagnose which motors are actually responding
 * 
 * Tests if Motor 1 responds to all CAN IDs, or if other motors exist
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

static bool init_serial(void) {
    struct termios tty;
    
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
    
    return true;
}

static bool send_l91_command(const uint8_t* cmd, size_t len) {
    tcflush(serial_fd, TCIFLUSH);
    ssize_t written = write(serial_fd, cmd, len);
    if (written != len) return false;
    tcdrain(serial_fd);
    usleep(100000);  // 100ms
    return true;
}

/* Read any response from motor */
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

/* Deactivate motor */
static bool deactivate_motor(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x00, 0x07, 0xe8, can_id, 0x00, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

/* Activate motor */
static bool activate_motor(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x00, 0x07, 0xe8, can_id, 0x01, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

/* Load parameters */
static bool load_params(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x20, 0x07, 0xe8, can_id, 0x08, 0x00,
                     0xc4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

/* Move motor and check for response */
static bool test_motor_with_response(uint8_t can_id) {
    uint8_t response[256];
    int response_count = 0;
    
    // Clear any pending data
    tcflush(serial_fd, TCIFLUSH);
    
    // Deactivate all first
    for (uint8_t i = 0x01; i <= 0x0E; i++) {
        deactivate_motor(i);
    }
    usleep(300000);
    
    printf("  Testing CAN ID 0x%02X:\n", can_id);
    
    // Activate
    printf("    Activate: ");
    fflush(stdout);
    if (!activate_motor(can_id)) {
        printf("✗ Failed to send\n");
        return false;
    }
    
    // Check for response
    int got = read_response(response, sizeof(response), 500);
    if (got > 0) {
        printf("✓ Sent (got %d byte response)\n", got);
        response_count += got;
    } else {
        printf("✓ Sent (no response)\n");
    }
    
    usleep(200000);
    
    // Load params
    printf("    Load params: ");
    fflush(stdout);
    if (!load_params(can_id)) {
        printf("✗ Failed\n");
        return false;
    }
    
    got = read_response(response, sizeof(response), 500);
    if (got > 0) {
        printf("✓ Sent (got %d byte response)\n", got);
        response_count += got;
    } else {
        printf("✓ Sent (no response)\n");
    }
    
    usleep(200000);
    
    // Try a move command
    uint8_t move_cmd[] = {0x41, 0x54, 0x90, 0x07, 0xe8, can_id, 0x08, 0x05,
                          0x70, 0x00, 0x00, 0x07, 0x01, 0x80, 0xa3, 0x0d, 0x0a};
    
    printf("    Move command: ");
    fflush(stdout);
    if (!send_l91_command(move_cmd, sizeof(move_cmd))) {
        printf("✗ Failed\n");
    } else {
        got = read_response(response, sizeof(response), 500);
        if (got > 0) {
            printf("✓ Sent (got %d byte response)\n", got);
            response_count += got;
        } else {
            printf("✓ Sent (no response)\n");
        }
    }
    
    usleep(300000);
    
    // Stop
    uint8_t stop_cmd[] = {0x41, 0x54, 0x90, 0x07, 0xe8, can_id, 0x08, 0x05,
                          0x70, 0x00, 0x00, 0x07, 0x00, 0x7f, 0xff, 0x0d, 0x0a};
    send_l91_command(stop_cmd, sizeof(stop_cmd));
    
    // Deactivate
    deactivate_motor(can_id);
    
    printf("    Total responses: %d bytes\n", response_count);
    
    return response_count > 0;
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Motor Response Diagnostic                 ║\n");
    printf("║  Checking which motors actually respond   ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    if (!init_serial()) {
        fprintf(stderr, "❌ Failed to initialize serial port\n");
        return 1;
    }
    
    printf("🔍 Testing CAN IDs to see which motors respond\n");
    printf("   (Looking for serial responses, not just movement)\n");
    printf("\n");
    printf("═══════════════════════════════════════════\n\n");
    
    // Test key CAN IDs based on Motor Studio log
    uint8_t test_ids[] = {0x01, 0x05, 0x09, 0x0C, 0x0D, 0x0E};
    const char* names[] = {"Motor 1", "Motor 5", "Motor 9", "Motor 12", "Motor 13", "Motor 14"};
    
    int responding_count = 0;
    
    for (int i = 0; i < 6; i++) {
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("%s (CAN ID 0x%02X)\n", names[i], test_ids[i]);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        if (test_motor_with_response(test_ids[i])) {
            printf("  ✅ %s RESPONDED\n", names[i]);
            responding_count++;
        } else {
            printf("  ❌ %s NO RESPONSE\n", names[i]);
        }
        
        printf("\n");
        usleep(500000);  // 500ms between tests
    }
    
    printf("═══════════════════════════════════════════\n");
    printf("📊 Summary:\n");
    printf("═══════════════════════════════════════════\n");
    printf("Motors responding: %d out of 6 tested\n", responding_count);
    printf("\n");
    printf("💡 Key Questions:\n");
    printf("   1. Did you see Motor 1 move for ALL CAN IDs?\n");
    printf("   2. Or did different CAN IDs move different motors?\n");
    printf("   3. Which physical motor moved for each CAN ID?\n");
    printf("\n");
    
    close(serial_fd);
    return 0;
}

