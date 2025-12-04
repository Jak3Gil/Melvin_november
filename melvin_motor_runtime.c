/*
 * melvin_motor_runtime.c - Real-time Motor Control Runtime
 * 
 * Runs continuously on Jetson:
 * 1. Monitors brain's motor EXEC nodes
 * 2. When EXEC activates, sends CAN commands to motors
 * 3. Reads motor feedback and feeds back to brain
 * 4. Learns motor control patterns through experience
 * 
 * This is the bridge between brain's EXEC outputs and physical CAN motors
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <time.h>

#define MOTOR_EXEC_BASE 2200
#define MOTOR_FEEDBACK_BASE 200
#define MAX_MOTORS 14

#define CAN_INTERFACE "can0"

/* Motor state tracking */
typedef struct {
    uint8_t can_id;
    bool active;
    float last_command;
    float current_position;
    float current_velocity;
    float current_torque;
    uint64_t last_update_us;
} MotorState;

static MotorState motor_states[MAX_MOTORS];
static int can_socket = -1;
static bool running = true;
static Graph *brain = NULL;
static pthread_mutex_t can_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Signal handler */
static void signal_handler(int sig) {
    printf("\n🛑 Shutting down...\n");
    running = false;
}

/* Get current time in microseconds */
static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* Initialize CAN */
static bool init_can(void) {
    struct sockaddr_can addr;
    struct ifreq ifr;
    
    can_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (can_socket < 0) {
        perror("CAN socket");
        return false;
    }
    
    strcpy(ifr.ifr_name, CAN_INTERFACE);
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
    
    /* Set non-blocking */
    int flags = fcntl(can_socket, F_GETFL, 0);
    fcntl(can_socket, F_SETFL, flags | O_NONBLOCK);
    
    printf("✅ CAN initialized\n");
    return true;
}

/* Send motor command via CAN */
static bool send_motor_command(uint8_t motor_id, uint8_t can_id, float value) {
    struct can_frame frame;
    
    /* Convert value (0.0 - 1.0) to motor-specific command */
    /* This is simplified - real motor controllers have specific protocols */
    
    frame.can_id = can_id;
    frame.can_dlc = 8;
    
    /* Command byte */
    frame.data[0] = 0x01;  /* Position command */
    
    /* Position value (as 32-bit float) */
    memcpy(&frame.data[1], &value, sizeof(float));
    
    pthread_mutex_lock(&can_mutex);
    int ret = write(can_socket, &frame, sizeof(frame));
    pthread_mutex_unlock(&can_mutex);
    
    if (ret != sizeof(frame)) {
        return false;
    }
    
    /* Update state */
    motor_states[motor_id].last_command = value;
    motor_states[motor_id].last_update_us = get_time_us();
    
    return true;
}

/* Read motor feedback from CAN */
static bool read_motor_feedback(void) {
    struct can_frame frame;
    
    pthread_mutex_lock(&can_mutex);
    int ret = read(can_socket, &frame, sizeof(frame));
    pthread_mutex_unlock(&can_mutex);
    
    if (ret != sizeof(frame)) {
        return false;  /* No data available (non-blocking) */
    }
    
    /* Find which motor this is */
    for (int i = 0; i < MAX_MOTORS; i++) {
        if (motor_states[i].active && motor_states[i].can_id == frame.can_id) {
            /* Parse feedback (motor-specific protocol) */
            if (frame.data[0] == 0x10) {  /* Status message */
                memcpy(&motor_states[i].current_position, &frame.data[1], sizeof(float));
                motor_states[i].last_update_us = get_time_us();
                return true;
            }
        }
    }
    
    return false;
}

/* Monitor EXEC nodes and send motor commands */
static void monitor_exec_nodes(void) {
    static uint64_t last_check_us = 0;
    uint64_t now_us = get_time_us();
    
    /* Check every 10ms */
    if (now_us - last_check_us < 10000) {
        return;
    }
    last_check_us = now_us;
    
    /* Scan motor EXEC nodes */
    for (int motor_id = 0; motor_id < MAX_MOTORS; motor_id++) {
        if (!motor_states[motor_id].active) continue;
        
        uint32_t exec_id = MOTOR_EXEC_BASE + motor_id;
        if (exec_id >= brain->node_count) continue;
        
        Node *exec_node = &brain->nodes[exec_id];
        
        /* Check if EXEC wants to output (activation threshold) */
        if (exec_node->value > 0.5f && exec_node->node_type == NODE_TYPE_EXEC) {
            float command = exec_node->value;
            
            /* Only send if significantly different from last command */
            if (fabs(command - motor_states[motor_id].last_command) > 0.01f) {
                printf("🚀 Motor %d: Sending command %.3f (EXEC node %u)\n", 
                       motor_id, command, exec_id);
                
                if (send_motor_command(motor_id, motor_states[motor_id].can_id, command)) {
                    /* Reset activation to prevent repeated sends */
                    exec_node->value *= 0.9f;
                } else {
                    fprintf(stderr, "⚠️  Failed to send command to motor %d\n", motor_id);
                }
            }
        }
    }
}

/* Feed motor feedback back to brain */
static void feed_motor_feedback(void) {
    static uint64_t last_feed_us = 0;
    uint64_t now_us = get_time_us();
    
    /* Feed every 50ms */
    if (now_us - last_feed_us < 50000) {
        return;
    }
    last_feed_us = now_us;
    
    for (int motor_id = 0; motor_id < MAX_MOTORS; motor_id++) {
        if (!motor_states[motor_id].active) continue;
        
        uint32_t feedback_port = MOTOR_FEEDBACK_BASE + motor_id;
        if (feedback_port >= brain->node_count) continue;
        
        /* Feed position as activation value */
        Node *feedback_node = &brain->nodes[feedback_port];
        feedback_node->value = motor_states[motor_id].current_position;
        
        /* Also feed as string for pattern learning */
        char feedback_str[128];
        snprintf(feedback_str, sizeof(feedback_str), 
                "MOTOR_%d_POS_%.2f", motor_id, motor_states[motor_id].current_position);
        melvin_feed_string(brain, feedback_str);
    }
}

/* CAN receive thread */
static void *can_receive_thread(void *arg) {
    printf("📡 CAN receive thread started\n");
    
    while (running) {
        if (read_motor_feedback()) {
            /* Feedback received and processed */
        }
        usleep(1000);  /* 1ms */
    }
    
    return NULL;
}

/* Load motor configuration */
static int load_motor_config(const char *config_file) {
    FILE *f = fopen(config_file, "r");
    if (!f) {
        printf("⚠️  No motor config found, using defaults\n");
        /* Default configuration */
        for (int i = 0; i < MAX_MOTORS; i++) {
            motor_states[i].can_id = 0x01 + i;
            motor_states[i].active = false;  /* Will be detected */
        }
        return 0;
    }
    
    int count = 0;
    char line[256];
    int motor_id = -1;
    
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        
        if (sscanf(line, "motor_%d:", &motor_id) == 1) {
            if (motor_id >= 0 && motor_id < MAX_MOTORS) {
                motor_states[motor_id].active = true;
                count++;
            }
        } else if (motor_id >= 0 && motor_id < MAX_MOTORS) {
            unsigned int can_id;
            if (sscanf(line, "  can_id: 0x%X", &can_id) == 1) {
                motor_states[motor_id].can_id = (uint8_t)can_id;
            }
        }
    }
    
    fclose(f);
    printf("✅ Loaded config for %d motors\n", count);
    return count;
}

/* Main runtime loop */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <brain.m> [motor_config.txt]\n", argv[0]);
        fprintf(stderr, "\nThis runtime:\n");
        fprintf(stderr, "  - Monitors brain's motor EXEC nodes\n");
        fprintf(stderr, "  - Sends CAN commands when EXECs activate\n");
        fprintf(stderr, "  - Feeds motor state back to brain\n");
        fprintf(stderr, "  - Enables learned motor control\n");
        return 1;
    }
    
    printf("🤖 Melvin Motor Runtime\n");
    printf("========================\n\n");
    
    /* Setup signal handling */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Initialize motor states */
    memset(motor_states, 0, sizeof(motor_states));
    
    /* Load motor configuration */
    const char *config_file = argc > 2 ? argv[2] : "motor_config.txt";
    int motor_count = load_motor_config(config_file);
    
    /* Initialize CAN */
    if (!init_can()) {
        fprintf(stderr, "❌ Failed to initialize CAN\n");
        fprintf(stderr, "Make sure CAN is configured:\n");
        fprintf(stderr, "  sudo ip link set can0 type can bitrate 125000\n");
        fprintf(stderr, "  sudo ip link set can0 up\n");
        return 1;
    }
    
    /* Open brain */
    printf("Opening brain: %s\n", argv[1]);
    brain = melvin_open(argv[1], 100000, 50000000);
    if (!brain) {
        fprintf(stderr, "❌ Failed to open brain\n");
        close(can_socket);
        return 1;
    }
    
    printf("✅ Brain loaded (nodes: %u, edges: %u)\n", brain->node_count, brain->edge_count);
    
    /* Start CAN receive thread */
    pthread_t can_thread;
    if (pthread_create(&can_thread, NULL, can_receive_thread, NULL) != 0) {
        fprintf(stderr, "❌ Failed to start CAN thread\n");
        melvin_close(brain);
        close(can_socket);
        return 1;
    }
    
    printf("\n🚀 Motor runtime active!\n");
    printf("Monitoring %d motors...\n\n", motor_count);
    
    /* Main loop */
    uint64_t loop_count = 0;
    while (running) {
        /* Run UEL physics (pattern discovery + propagation) */
        melvin_call_entry(brain);
        
        /* Monitor EXEC nodes and send motor commands */
        monitor_exec_nodes();
        
        /* Feed motor feedback back to brain */
        feed_motor_feedback();
        
        /* Progress indicator */
        loop_count++;
        if (loop_count % 1000 == 0) {
            printf("⚙️  Loop %lu - Active motors: ", loop_count);
            for (int i = 0; i < MAX_MOTORS; i++) {
                if (motor_states[i].active) {
                    printf("%d(%.2f) ", i, motor_states[i].current_position);
                }
            }
            printf("\n");
        }
        
        usleep(1000);  /* 1ms loop time = 1kHz */
    }
    
    printf("\n🛑 Shutting down...\n");
    
    /* Stop all motors */
    for (int i = 0; i < MAX_MOTORS; i++) {
        if (motor_states[i].active) {
            send_motor_command(i, motor_states[i].can_id, 0.0f);
        }
    }
    
    /* Cleanup */
    pthread_join(can_thread, NULL);
    melvin_close(brain);
    close(can_socket);
    
    printf("✅ Shutdown complete\n");
    
    return 0;
}

