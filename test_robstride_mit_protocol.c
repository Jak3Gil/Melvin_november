/*
 * test_robstride_mit_protocol.c - Robstride O2 with MIT Protocol
 * 
 * Robstride O2 uses MIT Mini Cheetah CAN protocol
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

#define MOTOR_12  12
#define MOTOR_14  14

// MIT Protocol Commands
#define MIT_CMD_ENTER_CONTROL  0xFFFC  // Enter motor control mode
#define MIT_CMD_EXIT_CONTROL   0xFFFD  // Exit motor control mode
#define MIT_CMD_ZERO_POSITION  0xFFFE  // Set current position as zero

// Parameter limits (Robstride O2)
#define P_MIN  -12.5f
#define P_MAX   12.5f
#define V_MIN  -65.0f
#define V_MAX   65.0f
#define KP_MIN  0.0f
#define KP_MAX  500.0f
#define KD_MIN  0.0f
#define KD_MAX  5.0f
#define T_MIN  -18.0f
#define T_MAX   18.0f

static int can_socket = -1;
static bool running = true;

static void signal_handler(int sig) {
    printf("\n🛑 Stopping...\n");
    running = false;
}

static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("📡 Initializing CAN on slcan0...\n");
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("CAN socket");
        return false;
    }
    
    strcpy(ifr.ifr_name, "slcan0");
    if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
        perror("slcan0 not found");
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
    
    // Disable loopback and receive own messages
    int loopback = 0;
    setsockopt(can_socket, SOL_CAN_RAW, CAN_RAW_LOOPBACK, &loopback, sizeof(loopback));
    
    printf("✅ CAN ready (loopback disabled)\n\n");
    return true;
}

/* Pack float into uint16_t for MIT protocol */
static uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    
    return (uint16_t)((x - offset) * ((float)((1 << bits) - 1)) / span);
}

/* Unpack uint16_t to float from MIT protocol */
static float uint_to_float(uint16_t x_int, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    return ((float)x_int) * span / ((float)((1 << bits) - 1)) + offset;
}

/* Send MIT protocol control frame */
static bool send_mit_control(uint8_t motor_id, float pos, float vel, float kp, float kd, float torque) {
    struct can_frame frame;
    
    // Pack data according to MIT protocol
    uint16_t p_int = float_to_uint(pos, P_MIN, P_MAX, 16);
    uint16_t v_int = float_to_uint(vel, V_MIN, V_MAX, 12);
    uint16_t kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12);
    uint16_t kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12);
    uint16_t t_int = float_to_uint(torque, T_MIN, T_MAX, 12);
    
    frame.can_id = motor_id;
    frame.can_dlc = 8;
    
    // Pack according to MIT format:
    // Byte 0-1: Position (16 bits)
    // Byte 2-3: Velocity (12 bits) + Kp high (4 bits)
    // Byte 4-5: Kp low (8 bits) + Kd (12 bits) high (4 bits)
    // Byte 6-7: Kd low (8 bits) + Torque (12 bits) high (4 bits), Torque low (8 bits)
    
    frame.data[0] = (p_int >> 8) & 0xFF;
    frame.data[1] = p_int & 0xFF;
    frame.data[2] = (v_int >> 4) & 0xFF;
    frame.data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F);
    frame.data[4] = kp_int & 0xFF;
    frame.data[5] = (kd_int >> 4) & 0xFF;
    frame.data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F);
    frame.data[7] = t_int & 0xFF;
    
    // Clear RX buffer before sending
    struct can_frame rx_frame;
    while (read(can_socket, &rx_frame, sizeof(rx_frame)) > 0) {
        // Drain buffer
    }
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        perror("CAN write");
        return false;
    }
    
    // Small delay for motor to process
    usleep(1000);  // 1ms
    
    return true;
}

/* Enter motor control mode */
static bool enter_control_mode(uint8_t motor_id) {
    struct can_frame frame;
    
    printf("🔧 Entering control mode for motor %d... ", motor_id);
    fflush(stdout);
    
    frame.can_id = motor_id;
    frame.can_dlc = 8;
    
    // MIT enter control: 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFC
    memset(frame.data, 0xFF, 8);
    frame.data[7] = 0xFC;
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        printf("✗ Failed\n");
        return false;
    }
    
    usleep(100000);  // 100ms for motor to initialize
    printf("✓ Enabled\n");
    return true;
}

/* Exit motor control mode */
static bool exit_control_mode(uint8_t motor_id) {
    struct can_frame frame;
    
    printf("🔒 Exiting control mode for motor %d... ", motor_id);
    fflush(stdout);
    
    frame.can_id = motor_id;
    frame.can_dlc = 8;
    
    // MIT exit control: 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFF 0xFD
    memset(frame.data, 0xFF, 8);
    frame.data[7] = 0xFD;
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        printf("✗ Failed\n");
        return false;
    }
    
    usleep(50000);
    printf("✓ Disabled\n");
    return true;
}

/* Test motor with MIT protocol */
static void test_motor_mit(uint8_t motor_id, const char* name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing %s (MIT Protocol)\n", name);
    printf("═══════════════════════════════════════════\n\n");
    
    if (!enter_control_mode(motor_id)) {
        printf("❌ Failed to enter control mode\n\n");
        return;
    }
    
    sleep(1);
    
    printf("🔹 Slow position sweep with MIT protocol\n\n");
    
    // Use low kp for gentle movement
    float kp = 20.0f;
    float kd = 1.0f;
    
    for (float pos = -0.5f; pos <= 0.5f && running; pos += 0.2f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (send_mit_control(motor_id, pos, 0.0f, kp, kd, 0.0f)) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        usleep(500000);  // 500ms between commands
    }
    
    if (!running) {
        exit_control_mode(motor_id);
        return;
    }
    
    printf("\n🔹 Hold for 2 seconds\n\n");
    sleep(2);
    
    printf("🔹 Return to center\n\n");
    
    for (float pos = 0.5f; pos >= 0.0f && running; pos -= 0.2f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (send_mit_control(motor_id, pos, 0.0f, kp, kd, 0.0f)) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        usleep(500000);
    }
    
    printf("\n");
    exit_control_mode(motor_id);
    printf("✅ Test complete\n\n");
    sleep(1);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Robstride O2 Test (MIT Protocol)        ║\n");
    printf("║  Motors 12 & 14                          ║\n");
    printf("║  Press Ctrl+C to stop                    ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_can()) {
        fprintf(stderr, "❌ CAN init failed\n");
        fprintf(stderr, "Make sure slcan0 is up:\n");
        fprintf(stderr, "  sudo slcand -o -c -s6 /dev/ttyUSB0 slcan0\n");
        fprintf(stderr, "  sudo ip link set slcan0 up\n");
        return 1;
    }
    
    printf("⚠️  Ready to test with MIT protocol\n");
    printf("   This is the correct protocol for Robstride O2!\n");
    printf("\n");
    printf("Press Enter to test Motor 12...\n");
    getchar();
    
    if (running) test_motor_mit(MOTOR_12, "Motor 12");
    if (!running) goto cleanup;
    
    printf("Press Enter to test Motor 14...\n");
    getchar();
    
    if (running) test_motor_mit(MOTOR_14, "Motor 14");
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Test Complete!                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("📝 Did you see movement this time?\n");
    printf("   Motor 12 controls: __________________\n");
    printf("   Motor 14 controls: __________________\n");
    printf("\n");
    
cleanup:
    if (can_socket >= 0) {
        printf("🔒 Shutting down safely...\n");
        exit_control_mode(MOTOR_12);
        exit_control_mode(MOTOR_14);
        usleep(100000);
        close(can_socket);
    }
    
    printf("✅ Done\n");
    return 0;
}

