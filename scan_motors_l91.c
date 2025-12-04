/*
 * scan_motors_l91.c - Scan motors 1-15 with L91 protocol
 * 
 * Send commands to each motor and listen for responses
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

#define SERIAL_PORT "/dev/ttyUSB1"

static int serial_fd = -1;

static bool init_serial(void) {
    struct termios tty;
    
    printf("📡 Opening %s...\n", SERIAL_PORT);
    
    serial_fd = open(SERIAL_PORT, O_RDWR | O_NOCTTY);
    if (serial_fd < 0) {
        perror("open");
        return false;
    }
    
    if (tcgetattr(serial_fd, &tty) != 0) {
        perror("tcgetattr");
        close(serial_fd);
        return false;
    }
    
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    
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
    
    tty.c_cc[VTIME] = 5;
    tty.c_cc[VMIN] = 0;
    
    if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        close(serial_fd);
        return false;
    }
    
    printf("✅ Serial ready\n\n");
    return true;
}

static void send_and_listen(uint8_t can_id) {
    // Build L91 MOVE_JOG command
    uint8_t cmd[20];
    int idx = 0;
    
    cmd[idx++] = 0x41;  // 'A'
    cmd[idx++] = 0x54;  // 'T'
    cmd[idx++] = 0x90;  // Flags
    cmd[idx++] = 0x07;  // ID high
    cmd[idx++] = 0xe8;  // ID low
    cmd[idx++] = can_id;  // Motor CAN ID
    cmd[idx++] = 0x08;  // Length
    cmd[idx++] = 0x05;  // Command high
    cmd[idx++] = 0x70;  // Command low (MOVE_JOG)
    cmd[idx++] = 0x00;
    cmd[idx++] = 0x00;
    cmd[idx++] = 0x07;
    cmd[idx++] = 0x01;  // Flag = move
    cmd[idx++] = 0x80;  // Speed high (0x80A3 = medium speed)
    cmd[idx++] = 0xA3;  // Speed low
    cmd[idx++] = 0x0d;  // \r
    cmd[idx++] = 0x0a;  // \n
    
    printf("  Motor %2d (0x%02X): Sending MOVE_JOG... ", can_id, can_id);
    fflush(stdout);
    
    // Flush input
    tcflush(serial_fd, TCIFLUSH);
    
    // Send command
    if (write(serial_fd, cmd, idx) != idx) {
        printf("✗ Send failed\n");
        return;
    }
    
    // Wait for response
    uint8_t response[256];
    usleep(100000);  // 100ms
    
    int len = read(serial_fd, response, sizeof(response));
    
    if (len > 0) {
        printf("✅ Response (%d bytes): ", len);
        for (int i = 0; i < len && i < 20; i++) {
            printf("%02X ", response[i]);
        }
        printf("\n");
    } else {
        printf("⏱️  No response\n");
    }
    
    // Send stop
    cmd[12] = 0x00;  // Flag = stop
    cmd[13] = 0x7f;  // Speed = 0
    cmd[14] = 0xff;
    write(serial_fd, cmd, idx);
    usleep(50000);
    
    // Clear any responses
    while (read(serial_fd, response, sizeof(response)) > 0);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  L91 Motor Scanner (IDs 1-15)            ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    if (!init_serial()) {
        return 1;
    }
    
    printf("Scanning motor IDs 1-15...\n");
    printf("═══════════════════════════════════════════\n\n");
    
    for (uint8_t motor_id = 1; motor_id <= 15; motor_id++) {
        send_and_listen(motor_id);
        usleep(200000);  // 200ms between motors
    }
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Scan Complete                            ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("Check which motors responded above.\n");
    printf("Watch the robot - which motors moved?\n");
    printf("\n");
    
    if (serial_fd >= 0) {
        close(serial_fd);
    }
    
    return 0;
}

