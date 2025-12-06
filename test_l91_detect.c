/*
 * test_l91_detect.c - Test L91 device detection
 * 
 * Based on Motor Studio log:
 * 1. AT+AT -> detects devices
 * 2. AT 00 07e8 0c 01 00 -> activate device
 * 3. AT 20 07e8 0c 08 -> load params
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

static int serial_fd = -1;

static bool init_serial(void) {
    struct termios tty;
    
    printf("📡 Opening /dev/ttyUSB0...\n");
    
    serial_fd = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY);
    if (serial_fd < 0) {
        perror("open");
        return false;
    }
    
    tcgetattr(serial_fd, &tty);
    
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    
    tty.c_cflag = CS8 | CREAD | CLOCAL;
    tty.c_lflag = 0;
    tty.c_iflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VTIME] = 10;
    tty.c_cc[VMIN] = 0;
    
    tcsetattr(serial_fd, TCSANOW, &tty);
    
    printf("✅ Serial ready\n\n");
    return true;
}

static void send_hex(const char* name, const uint8_t* data, int len) {
    printf("📤 %s: ", name);
    for (int i = 0; i < len; i++) {
        printf("%02X ", data[i]);
    }
    printf("\n");
    
    tcflush(serial_fd, TCIOFLUSH);
    write(serial_fd, data, len);
    tcdrain(serial_fd);
    
    printf("   Sent. Waiting for response...\n");
    
    uint8_t response[512];
    usleep(500000);  // 500ms wait
    
    int got = read(serial_fd, response, sizeof(response));
    
    if (got > 0) {
        printf("   ✅ Response (%d bytes): ", got);
        for (int i = 0; i < got && i < 100; i++) {
            printf("%02X ", response[i]);
            if ((i + 1) % 16 == 0) printf("\n                           ");
        }
        printf("\n");
        
        // Also show as ASCII
        printf("   ASCII: ");
        for (int i = 0; i < got && i < 100; i++) {
            printf("%c", (response[i] >= 32 && response[i] < 127) ? response[i] : '.');
        }
        printf("\n");
    } else {
        printf("   ❌ No response\n");
    }
    
    printf("\n");
}

int main(void) {
    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║  L91 Detection Test                       ║\n");
    printf("║  Following Motor Studio sequence          ║\n");
    printf("╚═══════════════════════════════════════════╝\n\n");
    
    if (!init_serial()) {
        return 1;
    }
    
    printf("Following exact Motor Studio sequence...\n");
    printf("═══════════════════════════════════════════\n\n");
    
    // 1. Detection command: AT+AT\r\n (from line 1 of log)
    uint8_t detect[] = {0x41, 0x54, 0x2b, 0x41, 0x54, 0x0d, 0x0a};
    send_hex("1. AT+AT (detect)", detect, sizeof(detect));
    
    sleep(1);
    
    // 2. Activate device commands (from lines 2-17 of log)
    // AT 00 07 e8 0c 01 00 \r\n
    uint8_t activate12[] = {0x41, 0x54, 0x00, 0x07, 0xe8, 0x0c, 0x01, 0x00, 0x0d, 0x0a};
    send_hex("2. Activate motor 12", activate12, sizeof(activate12));
    
    // AT 00 07 e8 0e 01 00 \r\n (motor 14)
    uint8_t activate14[] = {0x41, 0x54, 0x00, 0x07, 0xe8, 0x0e, 0x01, 0x00, 0x0d, 0x0a};
    send_hex("3. Activate motor 14", activate14, sizeof(activate14));
    
    // 3. Load params (from line 4 of log)
    // AT 20 07 e8 0c 08 00 c4 00 00 00 00 00 00 \r\n
    uint8_t load_param12[] = {0x41, 0x54, 0x20, 0x07, 0xe8, 0x0c, 0x08, 0x00, 
                              0xc4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x0a};
    send_hex("4. Load params motor 12", load_param12, sizeof(load_param12));
    
    // 4. Move command (from line 23 of log)
    // AT 90 07 e8 0c 08 05 70 00 00 07 01 80 a3 \r\n
    uint8_t move12[] = {0x41, 0x54, 0x90, 0x07, 0xe8, 0x0c, 0x08, 0x05,
                        0x70, 0x00, 0x00, 0x07, 0x01, 0x80, 0xa3, 0x0d, 0x0a};
    send_hex("5. MOVE motor 12", move12, sizeof(move12));
    
    printf("═══════════════════════════════════════════\n");
    printf("Did motor 12 move?\n");
    printf("═══════════════════════════════════════════\n\n");
    
    // Stop motor 12
    uint8_t stop12[] = {0x41, 0x54, 0x90, 0x07, 0xe8, 0x0c, 0x08, 0x05,
                        0x70, 0x00, 0x00, 0x07, 0x00, 0x7f, 0xff, 0x0d, 0x0a};
    send_hex("6. STOP motor 12", stop12, sizeof(stop12));
    
    sleep(1);
    
    // Now motor 14
    uint8_t move14[] = {0x41, 0x54, 0x90, 0x07, 0xe8, 0x0e, 0x08, 0x05,
                        0x70, 0x00, 0x00, 0x07, 0x01, 0x80, 0xa3, 0x0d, 0x0a};
    send_hex("7. MOVE motor 14", move14, sizeof(move14));
    
    printf("═══════════════════════════════════════════\n");
    printf("Did motor 14 move?\n");
    printf("═══════════════════════════════════════════\n\n");
    
    uint8_t stop14[] = {0x41, 0x54, 0x90, 0x07, 0xe8, 0x0e, 0x08, 0x05,
                        0x70, 0x00, 0x00, 0x07, 0x00, 0x7f, 0xff, 0x0d, 0x0a};
    send_hex("8. STOP motor 14", stop14, sizeof(stop14));
    
    printf("\n✅ Test complete\n");
    
    close(serial_fd);
    return 0;
}

