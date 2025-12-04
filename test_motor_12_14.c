/*
 * test_motor_12_14.c - Test Motors 12 and 14 Slowly
 * 
 * Safely tests motors 12 and 14 with slow, controlled movements
 * to observe what physical components they control
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <signal.h>
#include <stdbool.h>

#define CAN_INTERFACE "can0"

static int can_socket = -1;
static bool running = true;

/* Signal handler for Ctrl+C */
static void signal_handler(int sig) {
    printf("\n\n🛑 Stopping motors safely...\n");
    running = false;
}

/* Initialize CAN */
static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("📡 Initializing CAN interface: %s\n", CAN_INTERFACE);
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("Failed to create CAN socket");
        return false;
    }
    
    strcpy(ifr.ifr_name, CAN_INTERFACE);
    if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
        perror("Failed to get CAN interface");
        printf("\nMake sure CAN is configured:\n");
        printf("  sudo ip link set can0 type can bitrate 125000\n");
        printf("  sudo ip link set can0 up\n");
        close(can_socket);
        return false;
    }
    
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    
    if (bind(can_socket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Failed to bind CAN socket");
        close(can_socket);
        return false;
    }
    
    printf("✅ CAN interface ready\n\n");
    return true;
}

/* Send position command to motor */
static bool send_motor_position(uint8_t motor_id, float position) {
    struct can_frame frame;
    
    /* Motor 12 = CAN ID 0x0D, Motor 14 = CAN ID 0x0F */
    /* Adjust based on actual motor mapping */
    uint8_t can_id = 0x01 + motor_id;
    
    frame.can_id = can_id;
    frame.can_dlc = 8;
    
    /* Position control command (motor-specific protocol) */
    frame.data[0] = 0x01;  /* Position mode */
    
    /* Pack position as float (0.0 to 1.0) */
    memcpy(&frame.data[1], &position, sizeof(float));
    
    /* Safety limits */
    frame.data[5] = 0x00;  /* Low speed */
    frame.data[6] = 0x00;
    frame.data[7] = 0x00;
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        perror("Failed to send CAN frame");
        return false;
    }
    
    return true;
}

/* Stop motor */
static bool stop_motor(uint8_t motor_id) {
    struct can_frame frame;
    uint8_t can_id = 0x01 + motor_id;
    
    frame.can_id = can_id;
    frame.can_dlc = 2;
    frame.data[0] = 0x05;  /* Disable/stop command */
    frame.data[1] = 0x00;
    
    write(can_socket, &frame, sizeof(frame));
    return true;
}

/* Test motor with slow movement */
static void test_motor_slow(uint8_t motor_id, const char *name) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing Motor %d (%s)\n", motor_id, name);
    printf("═══════════════════════════════════════════\n\n");
    
    printf("🔹 Phase 1: Slow movement from 0%% → 50%%\n");
    printf("   Watch what moves on the robot!\n\n");
    
    for (float pos = 0.0f; pos <= 0.5f && running; pos += 0.05f) {
        printf("   Position: %.0f%% ", pos * 100);
        fflush(stdout);
        
        if (send_motor_position(motor_id, pos)) {
            printf("✓ Sent\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);  /* 500ms delay - very slow */
    }
    
    if (!running) return;
    
    printf("\n🔹 Phase 2: Hold at 50%% for 2 seconds\n");
    printf("   Observe the position...\n\n");
    sleep(2);
    
    printf("🔹 Phase 3: Slow return to 0%%\n\n");
    
    for (float pos = 0.5f; pos >= 0.0f && running; pos -= 0.05f) {
        printf("   Position: %.0f%% ", pos * 100);
        fflush(stdout);
        
        if (send_motor_position(motor_id, pos)) {
            printf("✓ Sent\n");
        } else {
            printf("✗ Failed\n");
        }
        
        usleep(500000);  /* 500ms delay */
    }
    
    printf("\n🔹 Phase 4: Stop and disable motor\n");
    stop_motor(motor_id);
    
    printf("✅ Test complete for Motor %d\n\n", motor_id);
    sleep(1);
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║   Motor 12 & 14 Test - Slow Movement     ║\n");
    printf("║   Press Ctrl+C at any time to stop       ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    /* Setup signal handler */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Initialize CAN */
    if (!init_can()) {
        fprintf(stderr, "❌ Failed to initialize CAN\n");
        return 1;
    }
    
    printf("⚠️  SAFETY CHECK:\n");
    printf("   • Is the robot in a safe position?\n");
    printf("   • Are there obstacles in the way?\n");
    printf("   • Is someone ready to hit E-stop?\n");
    printf("\n");
    printf("Press Enter to start testing Motor 12...\n");
    getchar();
    
    if (!running) goto cleanup;
    
    /* Test Motor 12 */
    test_motor_slow(12, "Motor 12");
    
    if (!running) goto cleanup;
    
    printf("\n");
    printf("Press Enter to continue with Motor 14...\n");
    getchar();
    
    if (!running) goto cleanup;
    
    /* Test Motor 14 */
    test_motor_slow(14, "Motor 14");
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Test Complete!                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("What did you observe?\n");
    printf("  Motor 12: _______________________________\n");
    printf("  Motor 14: _______________________________\n");
    printf("\n");
    printf("Document your findings in motor_config.txt\n");
    
cleanup:
    if (can_socket >= 0) {
        /* Ensure motors are stopped */
        printf("\n🔒 Ensuring all motors stopped...\n");
        stop_motor(12);
        stop_motor(14);
        usleep(100000);
        close(can_socket);
    }
    
    printf("✅ Safe shutdown complete\n");
    return 0;
}

