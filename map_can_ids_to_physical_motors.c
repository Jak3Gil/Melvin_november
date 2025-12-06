/*
 * map_can_ids_to_physical_motors.c - Map CAN IDs to physical motors
 * 
 * Tests each responding CAN ID and asks which physical motor moved
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

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
    usleep(100000);
    return true;
}

static bool deactivate_motor(uint8_t can_id) {
    uint8_t cmd[] = {0x41, 0x54, 0x00, 0x07, 0xe8, can_id, 0x00, 0x00, 0x0d, 0x0a};
    return send_l91_command(cmd, sizeof(cmd));
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

static bool move_motor_jog(uint8_t can_id, float speed, uint8_t flag) {
    uint8_t cmd[20];
    int idx = 0;
    
    cmd[idx++] = 0x41;
    cmd[idx++] = 0x54;
    cmd[idx++] = 0x90;
    cmd[idx++] = 0x07;
    cmd[idx++] = 0xe8;
    cmd[idx++] = can_id;
    cmd[idx++] = 0x08;
    cmd[idx++] = 0x05;
    cmd[idx++] = 0x70;
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

static void test_and_identify(uint8_t can_id) {
    // Deactivate all
    for (uint8_t i = 0x01; i <= 0x0E; i++) {
        deactivate_motor(i);
    }
    usleep(300000);
    
    printf("═══════════════════════════════════════════\n");
    printf("Testing CAN ID 0x%02X\n", can_id);
    printf("═══════════════════════════════════════════\n\n");
    
    printf("🔧 Activating... ");
    fflush(stdout);
    activate_motor(can_id);
    printf("✓\n");
    
    usleep(300000);
    
    printf("📋 Loading parameters... ");
    fflush(stdout);
    load_params(can_id);
    printf("✓\n");
    
    usleep(300000);
    
    printf("\n🔹 Moving forward (3 seconds)...\n");
    move_motor_jog(can_id, 0.05f, 1);
    sleep(3);
    move_motor_jog(can_id, 0.0f, 0);
    printf("   Stopped\n");
    
    sleep(1);
    
    printf("\n🔹 Moving backward (3 seconds)...\n");
    move_motor_jog(can_id, -0.05f, 1);
    sleep(3);
    move_motor_jog(can_id, 0.0f, 0);
    printf("   Stopped\n");
    
    deactivate_motor(can_id);
    
    printf("\n");
    printf("❓ Which physical motor moved?\n");
    printf("   (Enter motor number, or 'none' if nothing moved)\n");
    printf("   Answer: ");
    fflush(stdout);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  CAN ID to Physical Motor Mapping         ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("This will test each responding CAN ID\n");
    printf("and ask which physical motor moved.\n");
    printf("\n");
    
    if (!init_serial()) {
        fprintf(stderr, "❌ Failed to initialize serial port\n");
        return 1;
    }
    
    // Test the responding CAN IDs
    uint8_t test_ids[] = {0x09, 0x0C, 0x0D, 0x0E};
    const char* labels[] = {"CAN ID 0x09", "CAN ID 0x0C", "CAN ID 0x0D", "CAN ID 0x0E"};
    
    printf("Press Enter to start mapping...\n");
    getchar();
    
    for (int i = 0; i < 4; i++) {
        test_and_identify(test_ids[i]);
        
        char answer[100];
        if (fgets(answer, sizeof(answer), stdin) == NULL) break;
        
        printf("\n");
        printf("✅ Recorded: %s → %s", labels[i], answer);
        printf("\n");
        printf("Press Enter to continue to next CAN ID...\n");
        getchar();
    }
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Mapping Complete!               ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    close(serial_fd);
    return 0;
}

