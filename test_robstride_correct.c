/*
 * test_robstride_correct.c - Using CORRECT Robstride Protocol
 * 
 * From your working code: ENABLE=0xA1, DISABLE=0xA0, READ=0x92
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <signal.h>
#include <fcntl.h>

#define MOTOR_12  12
#define MOTOR_14  14

// CORRECT Robstride commands from your working code
#define CMD_DISABLE  0xA0
#define CMD_ENABLE   0xA1
#define CMD_READ     0x92

// Parameter ranges
#define POS_MIN     -12.5f
#define POS_MAX      12.5f
#define VEL_MAX      65.0f
#define KP_MIN       0.0f
#define KP_MAX     500.0f

static int can_socket = -1;
static bool running = true;

static void signal_handler(int sig) {
    printf("\n🛑 Stopping...\n");
    running = false;
}

static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("📡 Initializing CAN (Robstride Protocol)...\n");
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("socket");
        return false;
    }
    
    strcpy(ifr.ifr_name, "can0");  // Your working code uses can0
    if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
        perror("can0");
        close(can_socket);
        return false;
    }
    
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    
    if (bind(can_socket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(can_socket);
        return false;
    }
    
    // Non-blocking
    int flags = fcntl(can_socket, F_GETFL, 0);
    fcntl(can_socket, F_SETFL, flags | O_NONBLOCK);
    
    printf("✅ CAN ready on can0\n\n");
    return true;
}

static uint16_t float_to_uint(float x, float x_min, float x_max) {
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    
    float span = x_max - x_min;
    float normalized = (x - x_min) / span;
    return (uint16_t)(normalized * 65535.0f);
}

static bool send_frame(uint8_t motor_id, uint8_t command, const uint8_t* data, uint8_t len) {
    struct can_frame frame;
    
    // Clear RX buffer first
    struct can_frame rx;
    while (read(can_socket, &rx, sizeof(rx)) > 0);
    
    frame.can_id = motor_id;
    frame.can_dlc = len;
    frame.data[0] = command;
    
    for (int i = 0; i < len - 1 && i < 7; i++) {
        frame.data[i + 1] = data[i];
    }
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        return false;
    }
    
    usleep(5000);  // 5ms delay
    return true;
}

static bool enable_motor(uint8_t motor_id) {
    printf("🔧 Enabling motor %d (0x%02X)... ", motor_id, motor_id);
    fflush(stdout);
    
    uint8_t data[7] = {0};
    
    if (!send_frame(motor_id, CMD_ENABLE, data, 8)) {
        printf("✗\n");
        return false;
    }
    
    usleep(100000);
    printf("✓\n");
    return true;
}

static bool disable_motor(uint8_t motor_id) {
    printf("🔒 Disabling motor %d... ", motor_id);
    fflush(stdout);
    
    uint8_t data[7] = {0};
    send_frame(motor_id, CMD_DISABLE, data, 8);
    
    usleep(50000);
    printf("✓\n");
    return true;
}

static bool read_motor_state(uint8_t motor_id) {
    printf("📖 Reading motor %d state... ", motor_id);
    fflush(stdout);
    
    uint8_t data[7] = {0};
    
    if (!send_frame(motor_id, CMD_READ, data, 8)) {
        printf("✗\n");
        return false;
    }
    
    // Wait for response
    struct can_frame rx;
    fd_set readfds;
    struct timeval timeout;
    
    FD_ZERO(&readfds);
    FD_SET(can_socket, &readfds);
    timeout.tv_sec = 0;
    timeout.tv_usec = 500000;  // 500ms
    
    if (select(can_socket + 1, &readfds, NULL, NULL, &timeout) > 0) {
        if (read(can_socket, &rx, sizeof(rx)) > 0) {
            printf("✅ Response: ID=0x%03X Data=[", rx.can_id);
            for (int i = 0; i < rx.can_dlc; i++) {
                printf("%02X", rx.data[i]);
                if (i < rx.can_dlc - 1) printf(" ");
            }
            printf("]\n");
            return true;
        }
    }
    
    printf("⏱️  No response\n");
    return false;
}

static bool set_position(uint8_t motor_id, float pos, float vel, float kp) {
    if (pos < POS_MIN) pos = POS_MIN;
    if (pos > POS_MAX) pos = POS_MAX;
    if (vel < 0.0f) vel = 0.0f;
    if (vel > VEL_MAX) vel = VEL_MAX;
    if (kp < KP_MIN) kp = KP_MIN;
    if (kp > KP_MAX) kp = KP_MAX;
    
    uint8_t data[7];
    
    // Pack data (from your working code format)
    uint16_t p_int = float_to_uint(pos, POS_MIN, POS_MAX);
    uint16_t v_int = float_to_uint(vel, 0.0f, VEL_MAX);
    uint16_t kp_int = float_to_uint(kp, KP_MIN, KP_MAX);
    
    data[0] = (p_int >> 8) & 0xFF;
    data[1] = p_int & 0xFF;
    data[2] = (v_int >> 8) & 0xFF;
    data[3] = v_int & 0xFF;
    data[4] = (kp_int >> 8) & 0xFF;
    data[5] = kp_int & 0xFF;
    data[6] = 0;  // Kd
    
    return send_frame(motor_id, CMD_ENABLE, data, 8);  // Position uses ENABLE command
}

static void test_motor(uint8_t motor_id, const char* name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing %s (Robstride Protocol)\n", name);
    printf("═══════════════════════════════════════════\n\n");
    
    // Read state first
    read_motor_state(motor_id);
    sleep(1);
    
    // Enable
    if (!enable_motor(motor_id)) {
        printf("❌ Enable failed\n\n");
        return;
    }
    
    sleep(1);
    
    printf("\n🔹 Movement test (0 → 0.3 rad)\n\n");
    
    for (float pos = 0.0f; pos <= 0.3f && running; pos += 0.1f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (set_position(motor_id, pos, 5.0f, 50.0f)) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        sleep(1);  // 1 second between positions
    }
    
    if (!running) {
        disable_motor(motor_id);
        return;
    }
    
    printf("\n🔹 Return to zero\n\n");
    
    for (float pos = 0.3f; pos >= 0.0f && running; pos -= 0.1f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (set_position(motor_id, pos, 5.0f, 50.0f)) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        sleep(1);
    }
    
    printf("\n");
    disable_motor(motor_id);
    printf("✅ Complete\n\n");
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Robstride O2 - Correct Protocol         ║\n");
    printf("║  Commands: 0xA1=Enable, 0xA0=Disable      ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_can()) {
        return 1;
    }
    
    printf("Testing Motor 12...\n");
    getchar();
    
    if (running) test_motor(MOTOR_12, "Motor 12");
    if (!running) goto cleanup;
    
    printf("\nTesting Motor 14...\n");
    getchar();
    
    if (running) test_motor(MOTOR_14, "Motor 14");
    
    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║    Did you see movement?                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n\n");
    
cleanup:
    if (can_socket >= 0) {
        disable_motor(MOTOR_12);
        disable_motor(MOTOR_14);
        close(can_socket);
    }
    
    return 0;
}

