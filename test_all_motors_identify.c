/*
 * test_all_motors_identify.c - Test all responding CAN IDs and identify physical motors
 * 
 * Since all motors are powered, this will test each CAN ID and show which motor moves
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
    printf("\n🛑 Stopping all motors...\n");
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

static void test_can_id(uint8_t can_id, const char* label) {
    // Deactivate all motors first
    printf("\n🔒 Deactivating all motors...\n");
    for (uint8_t i = 0x01; i <= 0x0E; i++) {
        deactivate_motor(i);
    }
    usleep(500000);  // 500ms
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Testing %s (CAN ID 0x%02X)              ║\n", label, can_id);
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
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
    
    printf("\n");
    printf("🔹 WATCH CAREFULLY - Which motor moves?\n");
    printf("   Moving FORWARD for 2 seconds...\n");
    printf("\n");
    
    move_motor_jog(can_id, 0.1f, 1);  // Larger movement to see clearly
    sleep(2);
    move_motor_jog(can_id, 0.0f, 0);
    
    printf("   ✓ Stopped\n");
    sleep(1);
    
    printf("\n");
    printf("   Moving BACKWARD for 2 seconds...\n");
    printf("\n");
    
    move_motor_jog(can_id, -0.1f, 1);
    sleep(2);
    move_motor_jog(can_id, 0.0f, 0);
    
    printf("   ✓ Stopped\n");
    
    deactivate_motor(can_id);
    
    printf("\n");
    printf("═══════════════════════════════════════════\n");
    printf("❓ Which physical motor moved?\n");
    printf("   (Enter motor number 1-14, or 'none' if nothing moved)\n");
    printf("   Answer: ");
    fflush(stdout);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Identify Physical Motors by CAN ID       ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("All motors are powered.\n");
    printf("This test will identify which physical motor\n");
    printf("corresponds to each CAN ID.\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_serial()) {
        fprintf(stderr, "❌ Failed to initialize serial port\n");
        return 1;
    }
    
    // Test the responding CAN IDs
    uint8_t test_ids[] = {0x09, 0x0C, 0x0D, 0x0E};
    const char* labels[] = {"CAN ID 0x09", "CAN ID 0x0C", "CAN ID 0x0D", "CAN ID 0x0E"};
    
    printf("Press Enter to start testing...\n");
    getchar();
    
    char mapping[4][100];
    
    for (int i = 0; i < 4 && running; i++) {
        test_can_id(test_ids[i], labels[i]);
        
        if (fgets(mapping[i], sizeof(mapping[i]), stdin) == NULL) {
            break;
        }
        
        // Remove newline
        size_t len = strlen(mapping[i]);
        if (len > 0 && mapping[i][len-1] == '\n') {
            mapping[i][len-1] = '\0';
        }
        
        printf("\n");
        printf("✅ Recorded: %s → Physical Motor %s\n", labels[i], mapping[i]);
        printf("\n");
        
        if (i < 3) {
            printf("Press Enter to continue to next CAN ID...\n");
            getchar();
        }
    }
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Mapping Results                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    for (int i = 0; i < 4; i++) {
        printf("  %s → Physical Motor %s\n", labels[i], mapping[i]);
    }
    
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
    
    printf("✅ Test complete\n");
    return 0;
}

