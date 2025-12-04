/*
 * test_motor_response.c - Check if Robstride O2 motors respond
 * 
 * Listen for motor feedback to verify communication
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <fcntl.h>

#define MOTOR_12  12
#define MOTOR_14  14

static int can_socket = -1;

static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("📡 Initializing CAN...\n");
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("socket");
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
    
    // Set non-blocking for reads
    int flags = fcntl(can_socket, F_GETFL, 0);
    fcntl(can_socket, F_SETFL, flags | O_NONBLOCK);
    
    // Disable loopback
    int loopback = 0;
    setsockopt(can_socket, SOL_CAN_RAW, CAN_RAW_LOOPBACK, &loopback, sizeof(loopback));
    
    printf("✅ CAN ready\n\n");
    return true;
}

static void print_frame(const char* dir, const struct can_frame* frame) {
    printf("%s ID=0x%03X DLC=%d Data=[", dir, frame->can_id, frame->can_dlc);
    for (int i = 0; i < frame->can_dlc; i++) {
        printf("%02X", frame->data[i]);
        if (i < frame->can_dlc - 1) printf(" ");
    }
    printf("]\n");
}

static bool send_and_wait_response(uint8_t motor_id, const uint8_t* data, uint8_t len, int timeout_ms) {
    struct can_frame tx_frame, rx_frame;
    
    // Build frame
    tx_frame.can_id = motor_id;
    tx_frame.can_dlc = len;
    memcpy(tx_frame.data, data, len);
    
    printf("\n");
    print_frame("TX", &tx_frame);
    
    // Clear any old data
    while (read(can_socket, &rx_frame, sizeof(rx_frame)) > 0);
    
    // Send
    if (write(can_socket, &tx_frame, sizeof(tx_frame)) != sizeof(tx_frame)) {
        printf("❌ Send failed\n");
        return false;
    }
    
    // Wait for response using select
    struct timeval tv;
    fd_set readfds;
    
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    
    FD_ZERO(&readfds);
    FD_SET(can_socket, &readfds);
    
    int ret = select(can_socket + 1, &readfds, NULL, NULL, &tv);
    
    if (ret > 0) {
        // Data available
        while (read(can_socket, &rx_frame, sizeof(rx_frame)) > 0) {
            // Check if it's from our motor
            if (rx_frame.can_id == motor_id) {
                print_frame("RX", &rx_frame);
                printf("✅ Motor %d responded!\n", motor_id);
                return true;
            } else {
                print_frame("RX", &rx_frame);
            }
        }
    } else if (ret == 0) {
        printf("⏱️  Timeout - no response\n");
    } else {
        printf("❌ Select error\n");
    }
    
    return false;
}

static void test_motor_communication(uint8_t motor_id) {
    printf("═══════════════════════════════════════════\n");
    printf("Testing Motor %d Communication\n", motor_id);
    printf("═══════════════════════════════════════════\n");
    
    // Test 1: Enter control mode
    printf("\n1️⃣  Enter Control Mode (0xFF...0xFC)\n");
    uint8_t enter_cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC};
    bool responded = send_and_wait_response(motor_id, enter_cmd, 8, 500);
    
    if (!responded) {
        printf("   ⚠️  Motor %d did not respond to enter control\n", motor_id);
    }
    
    usleep(200000);  // 200ms
    
    // Test 2: Zero position command
    printf("\n2️⃣  Zero Position Command (0xFF...0xFE)\n");
    uint8_t zero_cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};
    responded = send_and_wait_response(motor_id, zero_cmd, 8, 500);
    
    if (!responded) {
        printf("   ⚠️  Motor %d did not respond to zero command\n", motor_id);
    }
    
    usleep(200000);
    
    // Test 3: MIT control command (position=0, low gains)
    printf("\n3️⃣  MIT Control Command (position=0)\n");
    uint8_t control_cmd[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    responded = send_and_wait_response(motor_id, control_cmd, 8, 500);
    
    if (!responded) {
        printf("   ⚠️  Motor %d did not respond to control command\n", motor_id);
    }
    
    usleep(200000);
    
    // Test 4: Exit control mode
    printf("\n4️⃣  Exit Control Mode (0xFF...0xFD)\n");
    uint8_t exit_cmd[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
    send_and_wait_response(motor_id, exit_cmd, 8, 500);
    
    printf("\n");
}

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Motor Response Test                      ║\n");
    printf("║  Checking if motors acknowledge commands ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    
    if (!init_can()) {
        return 1;
    }
    
    printf("🔍 This test will:\n");
    printf("   1. Send commands to motors\n");
    printf("   2. Wait for responses\n");
    printf("   3. Show what motors say back\n");
    printf("\n");
    printf("Press Enter to start...\n");
    getchar();
    
    test_motor_communication(MOTOR_12);
    
    printf("\nPress Enter to test Motor 14...\n");
    getchar();
    
    test_motor_communication(MOTOR_14);
    
    printf("\n");
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║           Analysis                        ║\n");
    printf("╚═══════════════════════════════════════════╝\n");
    printf("\n");
    printf("If motors didn't respond:\n");
    printf("  • Check power (LED on motor?)\n");
    printf("  • Check CAN IDs (maybe not 12/14?)\n");
    printf("  • Check wiring (CAN-H/CAN-L correct?)\n");
    printf("  • Try power cycling motors\n");
    printf("  • Check motor manual for init sequence\n");
    printf("\n");
    
    if (can_socket >= 0) {
        close(can_socket);
    }
    
    return 0;
}

