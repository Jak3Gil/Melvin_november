/*
 * test_motors_one_at_a_time.c - Test motors one at a time
 * 
 * Motor Studio says "Multiple devices connection is forbidden"
 * So we must activate/deactivate motors individually
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
static bool running = true;

static void signal_handler(int sig) {
    printf("\n🛑 Stopping...\n");
    running = false;
}

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
    usleep(50000);
    return true;
}

/* Deactivate motor (close connection) */
static bool deactivate_motor(uint8_t can_id) {
    // AT 00 07 e8 <can_id> 00 00 0d 0a (close)
    uint8_t cmd[] = {0x41, 0x54, 0x00, 0x07, 0xe8, can_id, 0x00, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

/* Activate motor (open connection) */
static bool activate_motor(uint8_t can_id) {
    // AT 00 07 e8 <can_id> 01 00 0d 0a (open)
    uint8_t cmd[] = {0x41, 0x54, 0x00, 0x07, 0xe8, can_id, 0x01, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

/* Load parameters */
static bool load_params(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x20, 0x07, 0xe8, can_id, 0x08, 0x00,
                     0xc4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
}

/* Move motor */
static bool move_motor_jog(uint8_t can_id, float speed, uint8_t flag) {
    uint8_t cmd[20];
    int idx = 0;
    
    cmd[idx++] = 0x41;  // 'A'
    cmd[idx++] = 0x54;  // 'T'
    cmd[idx++] = 0x90;  // Command
    cmd[idx++] = 0x07;  // Address high
    cmd[idx++] = 0xe8;  // Address low
    cmd[idx++] = can_id;  // CAN ID
    cmd[idx++] = 0x08;  // Length
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

/* Test a single motor - activate, test, deactivate */
static void test_single_motor(uint8_t can_id, const char* name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing %s (CAN ID 0x%02X)\n", name, can_id);
    printf("═══════════════════════════════════════════\n\n");
    
    // First, deactivate any previous motor
    printf("🔒 Deactivating any previous motor...\n");
    for (uint8_t i = 0x01; i <= 0x0E; i++) {
        deactivate_motor(i);
    }
    usleep(300000);
    
    // Activate this motor
    printf("🔧 Activating %s... ", name);
    fflush(stdout);
    if (activate_motor(can_id)) {
        printf("✓\n");
    } else {
        printf("✗\n");
        return;
    }
    
    usleep(300000);
    
    // Load parameters
    printf("📋 Loading parameters... ");
    fflush(stdout);
    if (load_params(can_id)) {
        printf("✓\n");
    } else {
        printf("✗\n");
    }
    
    usleep(300000);
    
    printf("\n🔹 Small forward movement (speed = 0.05)\n\n");
    
    for (int i = 0; i < 2 && running; i++) {
        printf("   Move %d: ", i + 1);
        fflush(stdout);
        
        if (move_motor_jog(can_id, 0.05f, 1)) {
            printf("✓ Sent\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);
        
        move_motor_jog(can_id, 0.0f, 0);
        printf("   Stop: ✓\n");
        usleep(500000);
    }
    
    if (!running) {
        deactivate_motor(can_id);
        return;
    }
    
    printf("\n🔹 Small backward movement (speed = -0.05)\n\n");
    
    for (int i = 0; i < 2 && running; i++) {
        printf("   Move %d: ", i + 1);
        fflush(stdout);
        
        if (move_motor_jog(can_id, -0.05f, 1)) {
            printf("✓ Sent\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);
        
        move_motor_jog(can_id, 0.0f, 0);
        printf("   Stop: ✓\n");
        usleep(500000);
    }
    
    // Deactivate before moving to next
    printf("\n🔒 Deactivating %s...\n", name);
    deactivate_motor(can_id);
    usleep(300000);
    
    printf("\n✅ Test complete for %s\n\n", name);
    printf("💡 Did you see %s move? (y/n): ", name);
    fflush(stdout);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Test Motors One at a Time                 ║\n");
    printf("║  (Motor Studio: Multiple devices forbidden)║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_serial()) {
        fprintf(stderr, "❌ Failed to initialize serial port\n");
        return 1;
    }
    
    printf("⚠️  Testing motors individually\n");
    printf("   (Only one motor can be active at a time)\n");
    printf("\n");
    
    // Test Motor 1 (CAN ID 0x01) - Motor Studio says "CAN ID: 1"
    printf("Press Enter to test Motor 1 (CAN ID 0x01)...\n");
    getchar();
    if (running) {
        test_single_motor(0x01, "Motor 1");
    }
    
    if (!running) goto cleanup;
    
    // Test Motor 12 (CAN ID 0x0C) - from hex log
    printf("\nPress Enter to test Motor 12 (CAN ID 0x0C)...\n");
    getchar();
    if (running) {
        test_single_motor(0x0C, "Motor 12");
    }
    
    if (!running) goto cleanup;
    
    // Test Motor 5 (CAN ID 0x05)
    printf("\nPress Enter to test Motor 5 (CAN ID 0x05)...\n");
    getchar();
    if (running) {
        test_single_motor(0x05, "Motor 5");
    }
    
    if (!running) goto cleanup;
    
    // Test Motor 13 (CAN ID 0x0D)
    printf("\nPress Enter to test Motor 13 (CAN ID 0x0D)...\n");
    getchar();
    if (running) {
        test_single_motor(0x0D, "Motor 13");
    }
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Test Complete!                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("📝 Summary:\n");
    printf("   Which motors actually moved?\n");
    printf("\n");
    
cleanup:
    if (serial_fd >= 0) {
        printf("🔒 Deactivating all motors...\n");
        for (uint8_t i = 0x01; i <= 0x0E; i++) {
            deactivate_motor(i);
            move_motor_jog(i, 0.0f, 0);
        }
        usleep(200000);
        close(serial_fd);
    }
    
    printf("✅ Safe shutdown complete\n");
    return 0;
}

