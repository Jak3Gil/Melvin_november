/*
 * map_can_motors.c - CAN Motor Discovery and Mapping for Melvin
 * 
 * This tool:
 * 1. Scans CAN bus for connected motors (14 total)
 * 2. Maps each motor to a unique port in the brain
 * 3. Creates EXEC nodes with ARM64 motor control code
 * 4. Establishes motor control patterns
 * 
 * Usage: ./map_can_motors brain.m
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>

/* Motor port assignments in the graph */
#define MOTOR_PORT_BASE 3100    /* Ports 3100-3113 for 14 motors */
#define MOTOR_EXEC_BASE 2200    /* EXEC nodes 2200-2213 for motor control */
#define MOTOR_FEEDBACK_BASE 200 /* Feedback ports 200-213 for motor state */

/* CAN configuration */
#define CAN_INTERFACE "can0"
#define CAN_BITRATE 125000
#define MAX_MOTORS 14

/* Motor IDs typically start at 0x01 */
#define MOTOR_ID_START 0x01

/* Motor control commands (typical CAN motor controller) */
#define CMD_SET_POSITION  0x01
#define CMD_SET_VELOCITY  0x02
#define CMD_SET_TORQUE    0x03
#define CMD_ENABLE        0x04
#define CMD_DISABLE       0x05
#define CMD_READ_STATE    0x10

/* Motor state structure */
typedef struct {
    uint8_t can_id;
    bool detected;
    char name[32];
    uint32_t port_id;         /* Output port for commands */
    uint32_t exec_id;         /* EXEC node for control code */
    uint32_t feedback_id;     /* Input port for state feedback */
} MotorInfo;

static MotorInfo motors[MAX_MOTORS];
static int can_socket = -1;

/* Initialize CAN interface */
static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    printf("Initializing CAN interface: %s\n", CAN_INTERFACE);
    
    /* Create socket */
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("Failed to create CAN socket");
        return false;
    }
    
    /* Get interface index */
    strcpy(ifr.ifr_name, CAN_INTERFACE);
    if (ioctl(can_socket, SIOCGIFINDEX, &ifr) < 0) {
        perror("Failed to get CAN interface index");
        printf("Make sure CAN is configured:\n");
        printf("  sudo ip link set can0 type can bitrate %d\n", CAN_BITRATE);
        printf("  sudo ip link set can0 up\n");
        close(can_socket);
        return false;
    }
    
    /* Bind socket to CAN interface */
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    
    if (bind(can_socket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Failed to bind CAN socket");
        close(can_socket);
        return false;
    }
    
    printf("✅ CAN interface ready\n");
    return true;
}

/* Send CAN message */
static bool send_can(uint8_t motor_id, uint8_t cmd, const uint8_t *data, uint8_t len) {
    struct can_frame frame;
    
    frame.can_id = motor_id;
    frame.can_dlc = len + 1;  /* Command + data */
    frame.data[0] = cmd;
    if (data && len > 0) {
        memcpy(&frame.data[1], data, len < 7 ? len : 7);
    }
    
    if (write(can_socket, &frame, sizeof(frame)) != sizeof(frame)) {
        return false;
    }
    
    return true;
}

/* Receive CAN message with timeout */
static bool recv_can(struct can_frame *frame, int timeout_ms) {
    fd_set readfds;
    struct timeval timeout;
    
    FD_ZERO(&readfds);
    FD_SET(can_socket, &readfds);
    
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;
    
    int ret = select(can_socket + 1, &readfds, NULL, NULL, &timeout);
    if (ret <= 0) return false;
    
    if (read(can_socket, frame, sizeof(*frame)) != sizeof(*frame)) {
        return false;
    }
    
    return true;
}

/* Scan CAN bus for motors */
static int scan_motors(void) {
    printf("\nScanning CAN bus for motors...\n");
    int detected = 0;
    
    for (int i = 0; i < MAX_MOTORS; i++) {
        uint8_t motor_id = MOTOR_ID_START + i;
        
        /* Try to ping motor */
        if (!send_can(motor_id, CMD_READ_STATE, NULL, 0)) {
            continue;
        }
        
        /* Wait for response */
        struct can_frame frame;
        if (recv_can(&frame, 100)) {  /* 100ms timeout */
            if (frame.can_id == motor_id) {
                motors[detected].can_id = motor_id;
                motors[detected].detected = true;
                motors[detected].port_id = MOTOR_PORT_BASE + detected;
                motors[detected].exec_id = MOTOR_EXEC_BASE + detected;
                motors[detected].feedback_id = MOTOR_FEEDBACK_BASE + detected;
                snprintf(motors[detected].name, sizeof(motors[detected].name), 
                         "MOTOR_%d", detected);
                
                printf("  ✅ Motor %d detected (CAN ID 0x%02X)\n", detected, motor_id);
                detected++;
            }
        }
        
        usleep(10000);  /* 10ms delay between scans */
    }
    
    printf("\nFound %d motors\n", detected);
    return detected;
}

/* ARM64 machine code for motor control operations */

/* This is a simplified ARM64 function that:
 * 1. Takes motor ID and command in registers
 * 2. Sends CAN message via syscall
 * 3. Returns status
 * 
 * Real implementation would use CAN syscalls
 */
static const uint8_t motor_control_code[] = {
    /* ARM64 function prologue */
    0xfd, 0x7b, 0xbf, 0xa9,  /* stp x29, x30, [sp, #-16]! */
    0xfd, 0x03, 0x00, 0x91,  /* mov x29, sp */
    
    /* Parameters: x0 = motor_id, x1 = command, x2 = value */
    
    /* Prepare CAN frame on stack */
    0xe0, 0x0f, 0x00, 0xf9,  /* str x0, [sp, #24]  ; can_id */
    0xe1, 0x13, 0x00, 0xf9,  /* str x1, [sp, #32]  ; command */
    0xe2, 0x17, 0x00, 0xf9,  /* str x2, [sp, #40]  ; data */
    
    /* Syscall to write CAN message */
    /* x0 = socket, x1 = frame ptr, x2 = frame size */
    0x60, 0x00, 0x80, 0xd2,  /* mov x0, #3 (hardcoded socket - would be loaded) */
    0xe1, 0x03, 0x00, 0x91,  /* mov x1, sp (frame pointer) */
    0x02, 0x02, 0x80, 0xd2,  /* mov x2, #16 (frame size) */
    0x08, 0x08, 0x80, 0xd2,  /* mov x8, #64 (SYS_write) */
    0x01, 0x00, 0x00, 0xd4,  /* svc #0 */
    
    /* Check return value */
    0x1f, 0x00, 0x00, 0x71,  /* cmp w0, #0 */
    0x60, 0x00, 0x9f, 0x1a,  /* cset w0, gt (return 1 if success) */
    
    /* Function epilogue */
    0xfd, 0x7b, 0xc1, 0xa8,  /* ldp x29, x30, [sp], #16 */
    0xc0, 0x03, 0x5f, 0xd6,  /* ret */
};

/* ARM64 code for reading motor position */
static const uint8_t motor_read_code[] = {
    /* Similar structure but reads instead of writes */
    0xfd, 0x7b, 0xbf, 0xa9,  /* stp x29, x30, [sp, #-16]! */
    0xfd, 0x03, 0x00, 0x91,  /* mov x29, sp */
    
    /* Send read command */
    0x00, 0x02, 0x80, 0xd2,  /* mov x0, #16 (CMD_READ_STATE) */
    /* ... send CAN message ... */
    
    /* Read response */
    0x60, 0x00, 0x80, 0xd2,  /* mov x0, #3 (socket) */
    0xe1, 0x03, 0x00, 0x91,  /* mov x1, sp */
    0x02, 0x02, 0x80, 0xd2,  /* mov x2, #16 */
    0x08, 0x08, 0x80, 0xd2,  /* mov x8, #63 (SYS_read) */
    0x01, 0x00, 0x00, 0xd4,  /* svc #0 */
    
    /* Return position from frame data */
    0xe0, 0x17, 0x40, 0xf9,  /* ldr x0, [sp, #40] */
    
    0xfd, 0x7b, 0xc1, 0xa8,  /* ldp x29, x30, [sp], #16 */
    0xc0, 0x03, 0x5f, 0xd6,  /* ret */
};

/* Teach motor control to brain */
static bool teach_motor_operations(Graph *g, int motor_count) {
    printf("\nTeaching motor operations to brain...\n");
    
    for (int i = 0; i < motor_count; i++) {
        MotorInfo *motor = &motors[i];
        
        /* Create EXEC node with motor control code */
        char label[128];
        snprintf(label, sizeof(label), "MOTOR_%d_CONTROL", i);
        
        /* Feed the EXEC node creation pattern */
        melvin_feed_string(g, label);
        melvin_call_entry(g);
        
        /* Find the created node */
        uint32_t exec_node = UINT32_MAX;
        for (uint32_t n = MOTOR_EXEC_BASE; n < MOTOR_EXEC_BASE + 100; n++) {
            if (n < g->node_count) {
                Node *node = &g->nodes[n];
                if (node->value > 0.1f) {
                    /* Check if label matches */
                    char node_label[128] = {0};
                    if (node->data_size > 0 && node->data_offset < g->blob_size) {
                        uint32_t copy_size = node->data_size < sizeof(node_label) - 1 ? 
                                           node->data_size : sizeof(node_label) - 1;
                        memcpy(node_label, g->blob + node->data_offset, copy_size);
                        
                        if (strcmp(node_label, label) == 0) {
                            exec_node = n;
                            break;
                        }
                    }
                }
            }
        }
        
        if (exec_node == UINT32_MAX) {
            /* Create new EXEC node */
            exec_node = motor->exec_id;
            if (exec_node >= g->node_count) {
                fprintf(stderr, "Error: Need to expand graph for motor %d\n", i);
                continue;
            }
            
            Node *node = &g->nodes[exec_node];
            node->value = 1.0f;
            node->node_type = NODE_TYPE_EXEC;
            
            /* Store motor control code in blob */
            if (g->blob_used + sizeof(motor_control_code) + strlen(label) + 1 < g->blob_size) {
                /* Store label */
                strcpy((char *)(g->blob + g->blob_used), label);
                node->data_offset = g->blob_used;
                node->data_size = strlen(label) + 1;
                g->blob_used += node->data_size;
                
                /* Store machine code */
                memcpy(g->blob + g->blob_used, motor_control_code, sizeof(motor_control_code));
                node->code_offset = g->blob_used;
                node->code_size = sizeof(motor_control_code);
                g->blob_used += sizeof(motor_control_code);
                
                /* Align to 8 bytes */
                g->blob_used = (g->blob_used + 7) & ~7;
                
                printf("  ✅ Taught motor %d control (EXEC node %u, code size %zu bytes)\n", 
                       i, exec_node, sizeof(motor_control_code));
            }
        }
        
        motor->exec_id = exec_node;
    }
    
    return true;
}

/* Create motor port patterns */
static bool create_motor_patterns(Graph *g, int motor_count) {
    printf("\nCreating motor port patterns...\n");
    
    for (int i = 0; i < motor_count; i++) {
        MotorInfo *motor = &motors[i];
        
        /* Create output port for motor commands */
        char port_label[64];
        snprintf(port_label, sizeof(port_label), "MOTOR_%d_CMD", i);
        melvin_feed_string(g, port_label);
        melvin_call_entry(g);
        
        /* Create feedback port for motor state */
        snprintf(port_label, sizeof(port_label), "MOTOR_%d_STATE", i);
        melvin_feed_string(g, port_label);
        melvin_call_entry(g);
        
        /* Create pattern for motor control */
        snprintf(port_label, sizeof(port_label), "MOVE_MOTOR_%d", i);
        melvin_feed_string(g, port_label);
        melvin_call_entry(g);
        
        printf("  ✅ Created patterns for motor %d\n", i);
    }
    
    return true;
}

/* Main */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <brain.m>\n", argv[0]);
        fprintf(stderr, "\nThis tool will:\n");
        fprintf(stderr, "  1. Scan CAN bus for motors\n");
        fprintf(stderr, "  2. Map each motor to brain ports\n");
        fprintf(stderr, "  3. Teach motor control code to EXEC nodes\n");
        fprintf(stderr, "  4. Create motor control patterns\n");
        fprintf(stderr, "\nMake sure CAN is configured first:\n");
        fprintf(stderr, "  sudo ip link set can0 type can bitrate 125000\n");
        fprintf(stderr, "  sudo ip link set can0 up\n");
        return 1;
    }
    
    printf("🤖 Melvin CAN Motor Mapper\n");
    printf("==========================\n\n");
    
    /* Initialize motors array */
    memset(motors, 0, sizeof(motors));
    
    /* Initialize CAN */
    if (!init_can()) {
        fprintf(stderr, "❌ Failed to initialize CAN\n");
        return 1;
    }
    
    /* Scan for motors */
    int motor_count = scan_motors();
    if (motor_count == 0) {
        fprintf(stderr, "❌ No motors found on CAN bus\n");
        fprintf(stderr, "Check connections and power\n");
        close(can_socket);
        return 1;
    }
    
    /* Open brain */
    printf("\nOpening brain: %s\n", argv[1]);
    struct stat st;
    if (stat(argv[1], &st) != 0) {
        fprintf(stderr, "❌ Brain file not found: %s\n", argv[1]);
        close(can_socket);
        return 1;
    }
    
    Graph *brain = melvin_open(argv[1], 100000, 50000000);
    if (!brain) {
        fprintf(stderr, "❌ Failed to open brain\n");
        close(can_socket);
        return 1;
    }
    
    printf("✅ Brain loaded (nodes: %u, edges: %u)\n", brain->node_count, brain->edge_count);
    
    /* Teach motor operations */
    if (!teach_motor_operations(brain, motor_count)) {
        fprintf(stderr, "❌ Failed to teach motor operations\n");
        melvin_close(brain);
        close(can_socket);
        return 1;
    }
    
    /* Create motor patterns */
    if (!create_motor_patterns(brain, motor_count)) {
        fprintf(stderr, "❌ Failed to create motor patterns\n");
        melvin_close(brain);
        close(can_socket);
        return 1;
    }
    
    /* Save motor configuration */
    FILE *config = fopen("motor_config.txt", "w");
    if (config) {
        fprintf(config, "# Melvin Motor Configuration\n");
        fprintf(config, "# Generated by map_can_motors\n\n");
        for (int i = 0; i < motor_count; i++) {
            fprintf(config, "motor_%d:\n", i);
            fprintf(config, "  name: %s\n", motors[i].name);
            fprintf(config, "  can_id: 0x%02X\n", motors[i].can_id);
            fprintf(config, "  port_id: %u\n", motors[i].port_id);
            fprintf(config, "  exec_id: %u\n", motors[i].exec_id);
            fprintf(config, "  feedback_id: %u\n", motors[i].feedback_id);
            fprintf(config, "\n");
        }
        fclose(config);
        printf("\n✅ Motor configuration saved to motor_config.txt\n");
    }
    
    /* Summary */
    printf("\n🎉 Motor Mapping Complete!\n");
    printf("===========================\n");
    printf("Motors mapped: %d\n", motor_count);
    printf("EXEC nodes: %u - %u\n", MOTOR_EXEC_BASE, MOTOR_EXEC_BASE + motor_count - 1);
    printf("Command ports: %u - %u\n", MOTOR_PORT_BASE, MOTOR_PORT_BASE + motor_count - 1);
    printf("Feedback ports: %u - %u\n", MOTOR_FEEDBACK_BASE, MOTOR_FEEDBACK_BASE + motor_count - 1);
    
    printf("\nNext steps:\n");
    printf("  1. Test motor control: ./test_motor_exec brain.m 0\n");
    printf("  2. Teach movement patterns to brain\n");
    printf("  3. Let brain learn motor control through experience!\n");
    
    melvin_close(brain);
    close(can_socket);
    
    return 0;
}

