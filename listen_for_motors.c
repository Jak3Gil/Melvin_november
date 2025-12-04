/*
 * listen_for_motors.c - Listen for ANY motor responses
 * 
 * Send command and capture ALL CAN traffic to see motor responses
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <fcntl.h>
#include <time.h>

#define MOTOR_12  12
#define MOTOR_14  14

static int can_socket = -1;

static bool init_can(const char* interface) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("📡 Opening %s...\n", interface);
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("socket");
        return false;
    }
    
    strcpy(ifr.ifr_name, interface);
    if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
        perror(interface);
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
    
    // Enable receive filter to see all traffic
    struct can_filter rfilter[1];
    rfilter[0].can_id = 0;
    rfilter[0].can_mask = 0;  // Receive everything
    setsockopt(can_socket, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter));
    
    // Keep loopback OFF to avoid seeing our own TX
    int loopback = 0;
    setsockopt(can_socket, SOL_CAN_RAW, CAN_RAW_LOOPBACK, &loopback, sizeof(loopback));
    
    // Non-blocking reads
    int flags = fcntl(can_socket, F_GETFL, 0);
    fcntl(can_socket, F_SETFL, flags | O_NONBLOCK);
    
    printf("✅ %s ready\n", interface);
    return true;
}

static void send_command(uint8_t motor_id, const uint8_t* data, uint8_t len) {
    struct can_frame frame;
    
    frame.can_id = motor_id;
    frame.can_dlc = len;
    memcpy(frame.data, data, len);
    
    printf("\n📤 SENDING to Motor %d (0x%02X): [", motor_id, motor_id);
    for (int i = 0; i < len; i++) {
        printf("%02X", data[i]);
        if (i < len - 1) printf(" ");
    }
    printf("]\n");
    
    if (write(can_socket, &frame, sizeof(frame)) < 0) {
        perror("   Write failed");
    } else {
        printf("   ✓ Sent\n");
    }
}

static void listen_for_responses(int duration_ms) {
    printf("\n📥 Listening for responses (%d ms)...\n", duration_ms);
    
    struct can_frame frame;
    int response_count = 0;
    
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        long elapsed_ms = (now.tv_sec - start.tv_sec) * 1000 + 
                         (now.tv_nsec - start.tv_nsec) / 1000000;
        
        if (elapsed_ms >= duration_ms) break;
        
        ssize_t nbytes = read(can_socket, &frame, sizeof(frame));
        
        if (nbytes > 0) {
            printf("   📬 ID=0x%03X DLC=%d Data=[", frame.can_id, frame.can_dlc);
            for (int i = 0; i < frame.can_dlc; i++) {
                printf("%02X", frame.data[i]);
                if (i < frame.can_dlc - 1) printf(" ");
            }
            printf("]\n");
            response_count++;
        } else {
            usleep(1000);  // 1ms
        }
    }
    
    if (response_count == 0) {
        printf("   ❌ No responses received\n");
    } else {
        printf("\n   ✅ Received %d frames\n", response_count);
    }
}

int main(int argc, char** argv) {
    const char* interface = argc > 1 ? argv[1] : "slcan0";
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Motor Response Listener                  ║\n");
    printf("║  Interface: %-30s║\n", interface);
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    if (!init_can(interface)) {
        return 1;
    }
    
    // Test sequence
    uint8_t enter_control[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};
    uint8_t zero_position[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    uint8_t exit_control[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
    
    printf("Testing Motor 12 (0x0C)...\n");
    printf("══════════════════════════════════════════\n");
    
    send_command(MOTOR_12, enter_control, 8);
    listen_for_responses(1000);
    
    send_command(MOTOR_12, zero_position, 8);
    listen_for_responses(1000);
    
    send_command(MOTOR_12, exit_control, 8);
    listen_for_responses(1000);
    
    printf("\n\nTesting Motor 14 (0x0E)...\n");
    printf("══════════════════════════════════════════\n");
    
    send_command(MOTOR_14, enter_control, 8);
    listen_for_responses(1000);
    
    send_command(MOTOR_14, zero_position, 8);
    listen_for_responses(1000);
    
    send_command(MOTOR_14, exit_control, 8);
    listen_for_responses(1000);
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Summary                                  ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("Check above for any 📬 responses.\n");
    printf("If you see responses with DIFFERENT data than\n");
    printf("what we sent, those are motor feedback frames!\n");
    printf("\n");
    
    close(can_socket);
    return 0;
}

