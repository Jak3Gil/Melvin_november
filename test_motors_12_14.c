/*
 * test_motors_12_14.c - Test Robstride Motors 12 & 14
 * 
 * Based on working Robstride protocol
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

// Motor IDs
#define MOTOR_12  12  // 0x0C
#define MOTOR_14  14  // 0x0E

// Robstride commands
#define CMD_DISABLE_MOTOR  0xA0
#define CMD_ENABLE_MOTOR   0xA1
#define CMD_POSITION_MODE  0xA1

// Parameter ranges
#define POS_MIN     -12.5f
#define POS_MAX      12.5f
#define VEL_MAX      65.0f
#define KP_MIN        0.0f
#define KP_MAX      500.0f

static int can_socket = -1;
static bool running = true;

static void signal_handler(int sig) {
    printf("\n🛑 Stopping...\n");
    running = false;
}

static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("📡 Initializing USB-to-CAN...\n");
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("CAN socket");
        return false;
    }
    
    // Use slcan0 (USB-based CAN) instead of can0 (native CAN)
    strcpy(ifr.ifr_name, "slcan0");
    if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
        // Fallback to can0 if slcan0 doesn't exist
        strcpy(ifr.ifr_name, "can0");
        if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
            perror("CAN interface (tried slcan0 and can0)");
            close(can_socket);
            return false;
        }
    }
    
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    
    if (bind(can_socket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("CAN bind");
        close(can_socket);
        return false;
    }
    
    printf("✅ USB-to-CAN ready (interface: %s)\n\n", ifr.ifr_name);
    return true;
}

static void float_to_can_bytes(float value, uint8_t* bytes, float min, float max) {
    if (value < min) value = min;
    if (value > max) value = max;
    
    float normalized = (value - min) / (max - min);
    uint16_t encoded = (uint16_t)(normalized * 65535.0f);
    
    bytes[0] = (encoded >> 8) & 0xFF;
    bytes[1] = encoded & 0xFF;
}

static bool send_can_frame(uint8_t motor_id, uint8_t command, const uint8_t* data, uint8_t len) {
    struct can_frame frame;
    
    frame.can_id = motor_id;
    frame.can_dlc = len;
    frame.data[0] = command;
    
    for (int i = 0; i < len - 1 && i < 7; i++) {
        frame.data[i + 1] = data[i];
    }
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        perror("CAN write");
        return false;
    }
    
    return true;
}

static bool enable_motor(uint8_t motor_id) {
    printf("🔧 Enabling motor %d (0x%02X)... ", motor_id, motor_id);
    fflush(stdout);
    
    uint8_t data[7] = {0};
    
    if (!send_can_frame(motor_id, CMD_ENABLE_MOTOR, data, 8)) {
        printf("✗ Failed\n");
        return false;
    }
    
    usleep(100000);
    printf("✓ Enabled\n");
    return true;
}

static bool disable_motor(uint8_t motor_id) {
    printf("🔒 Disabling motor %d... ", motor_id);
    fflush(stdout);
    
    uint8_t data[7] = {0};
    send_can_frame(motor_id, CMD_DISABLE_MOTOR, data, 8);
    
    usleep(100000);
    printf("✓ Disabled\n");
    return true;
}

static bool set_position(uint8_t motor_id, float position, float velocity, float kp) {
    if (position < POS_MIN) position = POS_MIN;
    if (position > POS_MAX) position = POS_MAX;
    if (velocity < 0.0f) velocity = 0.0f;
    if (velocity > VEL_MAX) velocity = VEL_MAX;
    if (kp < KP_MIN) kp = KP_MIN;
    if (kp > KP_MAX) kp = KP_MAX;
    
    uint8_t data[7] = {0};
    float_to_can_bytes(position, &data[0], POS_MIN, POS_MAX);
    float_to_can_bytes(velocity, &data[2], 0.0f, VEL_MAX);
    float_to_can_bytes(kp, &data[4], KP_MIN, KP_MAX);
    data[6] = 0;
    
    return send_can_frame(motor_id, CMD_POSITION_MODE, data, 8);
}

static void test_motor(uint8_t motor_id, const char* name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing %s (Motor %d, CAN ID 0x%02X)\n", name, motor_id, motor_id);
    printf("═══════════════════════════════════════════\n\n");
    
    if (!enable_motor(motor_id)) {
        printf("❌ Failed to enable motor\n\n");
        return;
    }
    
    sleep(1);
    
    printf("🔹 Slow sweep: -0.3 → 0 → +0.3 rad\n\n");
    
    for (float pos = -0.3f; pos <= 0.3f && running; pos += 0.1f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (set_position(motor_id, pos, 3.0f, 30.0f)) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        usleep(500000);  // 500ms
    }
    
    if (!running) {
        disable_motor(motor_id);
        return;
    }
    
    printf("\n🔹 Hold for 2 seconds\n\n");
    sleep(2);
    
    printf("🔹 Return to center\n\n");
    
    for (float pos = 0.3f; pos >= 0.0f && running; pos -= 0.1f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (set_position(motor_id, pos, 3.0f, 30.0f)) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        usleep(500000);
    }
    
    printf("\n");
    disable_motor(motor_id);
    printf("✅ Test complete\n\n");
    sleep(1);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Robstride Motors 12 & 14 Test           ║\n");
    printf("║  Watch carefully - note what moves!       ║\n");
    printf("║  Press Ctrl+C to stop                    ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_can()) {
        fprintf(stderr, "❌ CAN init failed\n");
        return 1;
    }
    
    printf("⚠️  Ready to test motors 12 and 14\n");
    printf("   Watch the robot and note what moves!\n");
    printf("\n");
    printf("Press Enter to test Motor 12...\n");
    getchar();
    
    if (running) test_motor(MOTOR_12, "Motor 12");
    if (!running) goto cleanup;
    
    printf("Press Enter to test Motor 14...\n");
    getchar();
    
    if (running) test_motor(MOTOR_14, "Motor 14");
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Test Complete!                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("📝 What did you observe?\n");
    printf("   Motor 12 controls: __________________\n");
    printf("   Motor 14 controls: __________________\n");
    printf("\n");
    
cleanup:
    if (can_socket >= 0) {
        printf("🔒 Shutting down safely...\n");
        disable_motor(MOTOR_12);
        disable_motor(MOTOR_14);
        usleep(100000);
        close(can_socket);
    }
    
    printf("✅ Done\n");
    return 0;
}

