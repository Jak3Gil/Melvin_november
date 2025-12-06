/*
 * test_motor_1_l91.c - Test Motor 1 using L91 protocol
 * 
 * Based on Motor Studio: "Detected device: CAN ID: 1"
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
#define MOTOR_1_CAN_ID 0x01  // Motor 1 = CAN ID 1

#ifdef B921600
#define BAUD_RATE B921600
#else
#define BAUD_RATE B115200
#define USE_CUSTOM_BAUD 921600
#endif

static int serial_fd = -1;
static bool running = true;

static void signal_handler(int sig) {
    printf("\n🛑 Stopping...\n");
    running = false;
}

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

static bool send_l91_command(const uint8_t* cmd, size_t len) {
    tcflush(serial_fd, TCIFLUSH);
    
    ssize_t written = write(serial_fd, cmd, len);
    if (written != len) {
        perror("Serial write");
        return false;
    }
    
    tcdrain(serial_fd);
    usleep(10000);
    
    return true;
}

static bool move_motor_jog(uint8_t can_id, float speed, uint8_t flag) {
    uint8_t cmd[20];
    int idx = 0;
    
    cmd[idx++] = 0x41;  // 'A'
    cmd[idx++] = 0x54;  // 'T'
    cmd[idx++] = 0x90;  // Command type
    cmd[idx++] = 0x07;  // Address high
    cmd[idx++] = 0xe8;  // Address low
    cmd[idx++] = can_id;  // Motor CAN ID
    cmd[idx++] = 0x08;  // Data length
    cmd[idx++] = 0x05;  // Command high
    cmd[idx++] = 0x70;  // Command low (MOVE_JOG)
    cmd[idx++] = 0x00;
    cmd[idx++] = 0x00;
    cmd[idx++] = 0x07;
    cmd[idx++] = flag;
    
    int16_t speed_val;
    if (speed == 0.0f) {
        speed_val = 0x7fff;
    } else if (speed > 0.0f) {
        speed_val = 0x8000 + (int16_t)(speed * 3283.0f);
    } else {
        speed_val = 0x7fff + (int16_t)(speed * 3283.0f);
    }
    
    cmd[idx++] = (speed_val >> 8) & 0xFF;
    cmd[idx++] = speed_val & 0xFF;
    cmd[idx++] = 0x0d;
    cmd[idx++] = 0x0a;
    
    return send_l91_command(cmd, idx);
}

static bool activate_motor(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x00, 0x07, 0xe8, can_id, 0x01, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

static bool load_params(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x20, 0x07, 0xe8, can_id, 0x08, 0x00,
                     0xc4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Testing Motor 1 (CAN ID 0x01)            ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_serial()) {
        return 1;
    }
    
    printf("Testing Motor 1 (CAN ID 0x01)...\n");
    printf("═══════════════════════════════════════════\n\n");
    
    printf("🔧 Activating Motor 1... ");
    fflush(stdout);
    if (activate_motor(MOTOR_1_CAN_ID)) {
        printf("✓\n");
    } else {
        printf("✗\n");
        return 1;
    }
    
    usleep(200000);
    
    printf("📋 Loading parameters... ");
    fflush(stdout);
    if (load_params(MOTOR_1_CAN_ID)) {
        printf("✓\n");
    } else {
        printf("✗\n");
    }
    
    usleep(200000);
    
    printf("\n🔹 Small forward movement (speed = 0.05)\n\n");
    
    for (int i = 0; i < 2 && running; i++) {
        printf("   Move %d: ", i + 1);
        fflush(stdout);
        
        if (move_motor_jog(MOTOR_1_CAN_ID, 0.05f, 1)) {
            printf("✓ Moving forward\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);
        move_motor_jog(MOTOR_1_CAN_ID, 0.0f, 0);
        printf("   Stop: ✓\n");
        usleep(500000);
    }
    
    if (!running) goto cleanup;
    
    printf("\n🔹 Small backward movement (speed = -0.05)\n\n");
    
    for (int i = 0; i < 2 && running; i++) {
        printf("   Move %d: ", i + 1);
        fflush(stdout);
        
        if (move_motor_jog(MOTOR_1_CAN_ID, -0.05f, 1)) {
            printf("✓ Moving backward\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);
        move_motor_jog(MOTOR_1_CAN_ID, 0.0f, 0);
        printf("   Stop: ✓\n");
        usleep(500000);
    }
    
    printf("\n✅ Test complete\n");
    printf("\nDid Motor 1 move? (This confirms CAN ID 0x01)\n");
    
cleanup:
    if (serial_fd >= 0) {
        move_motor_jog(MOTOR_1_CAN_ID, 0.0f, 0);
        usleep(100000);
        close(serial_fd);
    }
    
    return 0;
}

