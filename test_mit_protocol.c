/*
 * test_mit_protocol.c - Robstride O2 with MIT Mini Cheetah Protocol
 * 
 * Robstride O2 motors use MIT protocol with proper buffer management
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

/* Clear RX buffer - CRITICAL for Robstride! */
static void clear_rx_buffer(void) {
    struct can_frame frame;
    int flags = fcntl(can_socket, F_GETFL, 0);
    fcntl(can_socket, F_SETFL, flags | O_NONBLOCK);
    
    while (read(can_socket, &frame, sizeof(frame)) > 0) {
        // Drain all pending frames
    }
    
    fcntl(can_socket, F_SETFL, flags);
}

static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("📡 Initializing CAN (MIT Protocol)...\n");
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("CAN socket");
        return false;
    }
    
    strcpy(ifr.ifr_name, "slcan0");
    if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
        perror("slcan0");
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
    
    // Disable loopback
    int loopback = 0;
    setsockopt(can_socket, SOL_CAN_RAW, CAN_RAW_LOOPBACK, &loopback, sizeof(loopback));
    
    printf("✅ CAN ready\n\n");
    return true;
}

/* Pack float into uint for MIT protocol */
static uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    
    float span = x_max - x_min;
    return (uint16_t)((x - x_min) * ((float)((1 << bits) - 1)) / span);
}

/* Send MIT control command */
static bool send_mit_command(uint8_t motor_id, float pos, float vel, float kp, float kd, float torque) {
    struct can_frame frame;
    
    // Clear RX buffer FIRST
    clear_rx_buffer();
    
    // Pack MIT protocol data
    uint16_t p_int = float_to_uint(pos, P_MIN, P_MAX, 16);
    uint16_t v_int = float_to_uint(vel, V_MIN, V_MAX, 12);
    uint16_t kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12);
    uint16_t kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12);
    uint16_t t_int = float_to_uint(torque, T_MIN, T_MAX, 12);
    
    frame.can_id = motor_id;
    frame.can_dlc = 8;
    
    // MIT format packing
    frame.data[0] = (p_int >> 8) & 0xFF;
    frame.data[1] = p_int & 0xFF;
    frame.data[2] = (v_int >> 4) & 0xFF;
    frame.data[3] = ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F);
    frame.data[4] = kp_int & 0xFF;
    frame.data[5] = (kd_int >> 4) & 0xFF;
    frame.data[6] = ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F);
    frame.data[7] = t_int & 0xFF;
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        return false;
    }
    
    usleep(2000);  // 2ms delay for motor processing
    clear_rx_buffer();  // Clear after sending too
    
    return true;
}

/* Enter control mode (MIT protocol) */
static bool enter_control_mode(uint8_t motor_id) {
    struct can_frame frame;
    
    printf("🔧 Entering control mode (motor %d)... ", motor_id);
    fflush(stdout);
    
    clear_rx_buffer();
    
    frame.can_id = motor_id;
    frame.can_dlc = 8;
    memset(frame.data, 0xFF, 8);
    frame.data[7] = 0xFC;  // Enter control command
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        printf("✗ Failed\n");
        return false;
    }
    
    usleep(150000);  // 150ms for motor init
    clear_rx_buffer();
    printf("✓\n");
    return true;
}

/* Exit control mode */
static bool exit_control_mode(uint8_t motor_id) {
    struct can_frame frame;
    
    printf("🔒 Exiting control mode (motor %d)... ", motor_id);
    fflush(stdout);
    
    clear_rx_buffer();
    
    frame.can_id = motor_id;
    frame.can_dlc = 8;
    memset(frame.data, 0xFF, 8);
    frame.data[7] = 0xFD;  // Exit control command
    
    write(can_socket, &frame, sizeof(frame));
    usleep(100000);
    clear_rx_buffer();
    printf("✓\n");
    return true;
}

/* Test motor */
static void test_motor(uint8_t motor_id, const char* name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing %s (MIT Protocol)\n", name);
    printf("═══════════════════════════════════════════\n\n");
    
    if (!enter_control_mode(motor_id)) {
        printf("❌ Failed to enter control\n\n");
        return;
    }
    
    sleep(1);
    
    printf("🔹 Gentle movement test\n\n");
    
    float kp = 10.0f;   // Gentle stiffness
    float kd = 0.5f;     // Light damping
    
    for (float pos = 0.0f; pos <= 0.3f && running; pos += 0.1f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (send_mit_command(motor_id, pos, 0.0f, kp, kd, 0.0f)) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        usleep(500000);
    }
    
    if (!running) {
        exit_control_mode(motor_id);
        return;
    }
    
    printf("\n🔹 Return to zero\n\n");
    
    for (float pos = 0.3f; pos >= 0.0f && running; pos -= 0.1f) {
        printf("   Position: %+.2f rad ", pos);
        fflush(stdout);
        
        if (send_mit_command(motor_id, pos, 0.0f, kp, kd, 0.0f)) {
            printf("✓\n");
        } else {
            printf("✗\n");
        }
        
        usleep(500000);
    }
    
    printf("\n");
    exit_control_mode(motor_id);
    printf("✅ Complete\n\n");
    sleep(1);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Robstride O2 - MIT Protocol Test        ║\n");
    printf("║  With Buffer Management                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (!init_can()) {
        return 1;
    }
    
    printf("⚠️  MIT Protocol active - buffer clearing enabled\n");
    printf("   Motors should respond now!\n\n");
    printf("Press Enter for Motor 12...\n");
    getchar();
    
    if (running) test_motor(MOTOR_12, "Motor 12");
    if (!running) goto cleanup;
    
    printf("Press Enter for Motor 14...\n");
    getchar();
    
    if (running) test_motor(MOTOR_14, "Motor 14");
    
    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║    Did you see movement?                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n\n");
    
cleanup:
    if (can_socket >= 0) {
        exit_control_mode(MOTOR_12);
        exit_control_mode(MOTOR_14);
        close(can_socket);
    }
    
    return 0;
}

