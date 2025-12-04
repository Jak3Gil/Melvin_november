/*
 * test_robstride_motors.c - Test Robstride Motors 13 & 14
 * 
 * Based on working code from github.com/Jak3Gil/Melvin/core/motor
 * Robstride O2/O3 motor protocol
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

// Motor IDs (these ARE the CAN IDs for Robstride)
#define MOTOR_13  13  // 0x0D
#define MOTOR_14  14  // 0x0E

// Robstride CAN Commands
#define CMD_DISABLE_MOTOR  0xA0
#define CMD_ENABLE_MOTOR   0xA1
#define CMD_POSITION_MODE  0xA1
#define CMD_VELOCITY_MODE  0xA2
#define CMD_TORQUE_MODE    0xA3
#define CMD_READ_STATE     0x92
#define CMD_ZERO_POSITION  0x19

// Motor parameter ranges (from Robstride datasheet)
#define POS_MIN     -12.5f   // radians
#define POS_MAX      12.5f
#define VEL_MIN     -65.0f   // rad/s  
#define VEL_MAX      65.0f
#define TORQUE_MIN  -18.0f   // Nm
#define TORQUE_MAX   18.0f
#define KP_MIN        0.0f
#define KP_MAX      500.0f

static int can_socket = -1;
static bool running = true;

/* Signal handler */
static void signal_handler(int sig) {
    printf("\n🛑 Stopping...\n");
    running = false;
}

/* Initialize CAN */
static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("📡 Initializing CAN (Robstride protocol)...\n");
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("CAN socket");
        return false;
    }
    
    strcpy(ifr.ifr_name, "can0");
    if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
        perror("CAN interface");
        close(can_socket);
        return false;
    }
    
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    
    if (bind(can_socket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("CAN bind");
        close(can_socket);
        return false;
    }
    
    printf("✅ CAN ready (bitrate should be 500kbps)\n\n");
    return true;
}

/* Encode float to 16-bit CAN bytes (Robstride protocol) */
static void float_to_can_bytes(float value, uint8_t* bytes, float min, float max) {
    // Clamp to range
    if (value < min) value = min;
    if (value > max) value = max;
    
    // Normalize to 0-65535 range
    float normalized = (value - min) / (max - min);
    uint16_t encoded = (uint16_t)(normalized * 65535.0f);
    
    bytes[0] = (encoded >> 8) & 0xFF;  // High byte
    bytes[1] = encoded & 0xFF;          // Low byte
}

/* Send CAN frame (Robstride format) */
static bool send_can_frame(uint8_t motor_id, uint8_t command, const uint8_t* data, uint8_t len) {
    struct can_frame frame;
    
    frame.can_id = motor_id;  // Motor ID IS the CAN ID
    frame.can_dlc = len;
    frame.data[0] = command;   // Command in first byte
    
    // Copy data (up to 7 bytes after command)
    for (int i = 0; i < len - 1 && i < 7; i++) {
        frame.data[i + 1] = data[i];
    }
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        perror("CAN write");
        return false;
    }
    
    return true;
}

/* Enable motor */
static bool enable_motor(uint8_t motor_id) {
    printf("🔧 Enabling motor %d... ", motor_id);
    fflush(stdout);
    
    uint8_t data[7] = {0};
    
    if (!send_can_frame(motor_id, CMD_ENABLE_MOTOR, data, 8)) {
        printf("✗ Failed\n");
        return false;
    }
    
    usleep(50000);  // 50ms delay
    printf("✓ Enabled\n");
    return true;
}

/* Disable motor */
static bool disable_motor(uint8_t motor_id) {
    printf("🔒 Disabling motor %d... ", motor_id);
    fflush(stdout);
    
    uint8_t data[7] = {0};
    
    if (!send_can_frame(motor_id, CMD_DISABLE_MOTOR, data, 8)) {
        printf("✗ Failed\n");
        return false;
    }
    
    usleep(50000);  // 50ms delay
    printf("✓ Disabled\n");
    return true;
}

/* Set motor position (Robstride position control) */
static bool set_position(uint8_t motor_id, float position, float velocity, float kp) {
    // Clamp to safe ranges
    if (position < POS_MIN) position = POS_MIN;
    if (position > POS_MAX) position = POS_MAX;
    if (velocity < 0.0f) velocity = 0.0f;
    if (velocity > VEL_MAX) velocity = VEL_MAX;
    if (kp < KP_MIN) kp = KP_MIN;
    if (kp > KP_MAX) kp = KP_MAX;
    
    // Encode data (Robstride format)
    uint8_t data[7] = {0};
    float_to_can_bytes(position, &data[0], POS_MIN, POS_MAX);
    float_to_can_bytes(velocity, &data[2], 0.0f, VEL_MAX);
    float_to_can_bytes(kp, &data[4], KP_MIN, KP_MAX);
    data[6] = 0;  // Kd = 0
    
    return send_can_frame(motor_id, CMD_POSITION_MODE, data, 8);
}

/* Test motor with slow movement */
static void test_motor_slow(uint8_t motor_id, const char* name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing %s (Motor %d, CAN ID 0x%02X)\n", name, motor_id, motor_id);
    printf("═══════════════════════════════════════════\n\n");
    
    // Enable motor
    if (!enable_motor(motor_id)) {
        printf("❌ Failed to enable motor\n");
        return;
    }
    
    sleep(1);
    
    printf("🔹 Phase 1: Slow sweep -0.5 → 0 → +0.5 rad\n");
    printf("   (Using position control with kp=50.0)\n\n");
    
    // Sweep from -0.5 to +0.5 radians
    for (float pos = -0.5f; pos <= 0.5f && running; pos += 0.1f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (set_position(motor_id, pos, 5.0f, 50.0f)) {  // velocity=5.0, kp=50.0
            printf("✓ Sent\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);  // 500ms delay - very slow
    }
    
    if (!running) {
        disable_motor(motor_id);
        return;
    }
    
    printf("\n🔹 Phase 2: Hold at +0.5 rad for 2 seconds\n\n");
    sleep(2);
    
    printf("🔹 Phase 3: Return to 0 rad\n\n");
    
    for (float pos = 0.5f; pos >= 0.0f && running; pos -= 0.1f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (set_position(motor_id, pos, 5.0f, 50.0f)) {
            printf("✓ Sent\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);  // 500ms delay
    }
    
    printf("\n🔹 Phase 4: Disable motor\n");
    disable_motor(motor_id);
    
    printf("✅ Test complete for motor %d\n\n", motor_id);
    sleep(1);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Robstride Motor Test (Motors 13 & 14)   ║\n");
    printf("║  Protocol: Robstride O2/O3               ║\n");
    printf("║  Press Ctrl+C to stop                    ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_can()) {
        fprintf(stderr, "❌ CAN initialization failed\n");
        fprintf(stderr, "\nMake sure:\n");
        fprintf(stderr, "  sudo ip link set can0 type can bitrate 500000\n");
        fprintf(stderr, "  sudo ip link set can0 up\n");
        return 1;
    }
    
    printf("⚠️  SAFETY CHECK:\n");
    printf("   • Motors 13 and 14 ready?\n");
    printf("   • Clear workspace?\n");
    printf("   • E-stop accessible?\n");
    printf("\n");
    
    printf("Press Enter to test Motor 13...\n");
    getchar();
    
    if (!running) goto cleanup;
    
    test_motor_slow(MOTOR_13, "Motor 13");
    
    if (!running) goto cleanup;
    
    printf("Press Enter to test Motor 14...\n");
    getchar();
    
    if (!running) goto cleanup;
    
    test_motor_slow(MOTOR_14, "Motor 14");
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Test Complete!                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("What did you observe?\n");
    printf("  Motor 13: _______________________________\n");
    printf("  Motor 14: _______________________________\n");
    printf("\n");
    
cleanup:
    if (can_socket >= 0) {
        printf("🔒 Disabling all motors...\n");
        disable_motor(MOTOR_13);
        disable_motor(MOTOR_14);
        usleep(100000);
        close(can_socket);
    }
    
    printf("✅ Safe shutdown complete\n");
    return 0;
}

