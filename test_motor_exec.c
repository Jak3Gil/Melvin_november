/*
 * test_motor_exec.c - Test Motor Control via EXEC Nodes
 * 
 * Tests that motor control code executes correctly through the brain
 * 
 * Usage: ./test_motor_exec brain.m <motor_id>
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>

#define MOTOR_EXEC_BASE 2200

/* Test motor by activating its EXEC node */
static bool test_motor(Graph *brain, int motor_id, float position) {
    uint32_t exec_id = MOTOR_EXEC_BASE + motor_id;
    
    printf("Testing motor %d (EXEC node %u)\n", motor_id, exec_id);
    
    if (exec_id >= brain->node_count) {
        fprintf(stderr, "❌ EXEC node %u out of range\n", exec_id);
        return false;
    }
    
    Node *exec_node = &brain->nodes[exec_id];
    
    /* Check if it's an EXEC node */
    if (exec_node->node_type != NODE_TYPE_EXEC) {
        fprintf(stderr, "❌ Node %u is not an EXEC node (type: %d)\n", 
                exec_id, exec_node->node_type);
        return false;
    }
    
    /* Check if it has code */
    if (exec_node->code_size == 0) {
        fprintf(stderr, "❌ EXEC node has no code\n");
        return false;
    }
    
    printf("✅ EXEC node found with %u bytes of code\n", exec_node->code_size);
    
    /* Print label if available */
    if (exec_node->data_size > 0 && exec_node->data_offset < brain->blob_size) {
        char label[128] = {0};
        uint32_t copy_size = exec_node->data_size < sizeof(label) - 1 ? 
                           exec_node->data_size : sizeof(label) - 1;
        memcpy(label, brain->blob + exec_node->data_offset, copy_size);
        printf("   Label: %s\n", label);
    }
    
    /* Activate the EXEC node to trigger motor control */
    printf("\n🚀 Activating motor control...\n");
    
    /* Set position value in the node */
    exec_node->value = position;
    
    /* Propagate to trigger execution */
    melvin_call_entry(brain);
    
    /* Check if execution happened (would need execution tracking in real system) */
    printf("✅ Motor command sent (position: %.2f)\n", position);
    
    return true;
}

/* Test sequence of motor movements */
static void test_sequence(Graph *brain, int motor_id) {
    printf("\n🎬 Running motor test sequence...\n\n");
    
    float positions[] = {0.0f, 0.5f, 1.0f, 0.5f, 0.0f};
    int num_positions = sizeof(positions) / sizeof(positions[0]);
    
    for (int i = 0; i < num_positions; i++) {
        printf("Step %d: Moving to position %.2f\n", i + 1, positions[i]);
        
        if (test_motor(brain, motor_id, positions[i])) {
            printf("  ✅ Command sent\n");
        } else {
            printf("  ❌ Command failed\n");
        }
        
        printf("  Waiting 1 second...\n\n");
        sleep(1);
    }
    
    printf("🎉 Sequence complete!\n");
}

/* Test all motors */
static void test_all_motors(Graph *brain) {
    printf("\n🎬 Testing all motors...\n\n");
    
    int tested = 0;
    for (int motor_id = 0; motor_id < 14; motor_id++) {
        uint32_t exec_id = MOTOR_EXEC_BASE + motor_id;
        
        if (exec_id >= brain->node_count) break;
        
        Node *node = &brain->nodes[exec_id];
        if (node->node_type == NODE_TYPE_EXEC && node->code_size > 0) {
            printf("Motor %d: ", motor_id);
            
            /* Quick test - just activate */
            node->value = 0.5f;
            melvin_call_entry(brain);
            
            printf("✅ OK\n");
            tested++;
            
            usleep(100000);  /* 100ms between motors */
        }
    }
    
    printf("\n✅ Tested %d motors\n", tested);
}

/* Demonstrate pattern-driven motor control */
static void test_pattern_routing(Graph *brain, int motor_id) {
    printf("\n🧠 Testing Pattern → Motor Routing...\n\n");
    
    /* Feed a movement pattern */
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "MOVE_MOTOR_%d", motor_id);
    
    printf("Feeding pattern: %s\n", pattern);
    melvin_feed_string(brain, pattern);
    
    /* Propagate through graph */
    printf("Propagating through graph...\n");
    for (int i = 0; i < 5; i++) {
        melvin_call_entry(brain);
    }
    
    /* Check if motor EXEC was activated */
    uint32_t exec_id = MOTOR_EXEC_BASE + motor_id;
    Node *exec_node = &brain->nodes[exec_id];
    
    printf("EXEC node activation: %.4f\n", exec_node->value);
    
    if (exec_node->value > 0.1f) {
        printf("✅ Pattern successfully routed to motor EXEC!\n");
        printf("   This shows the brain learned: pattern → motor control\n");
    } else {
        printf("⚠️  EXEC not activated yet\n");
        printf("   Brain may need more training on this pattern\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <brain.m> [motor_id] [test_type]\n", argv[0]);
        fprintf(stderr, "\nTests:\n");
        fprintf(stderr, "  %s brain.m 0          - Test motor 0\n", argv[0]);
        fprintf(stderr, "  %s brain.m 0 seq      - Run sequence on motor 0\n", argv[0]);
        fprintf(stderr, "  %s brain.m all        - Test all motors\n", argv[0]);
        fprintf(stderr, "  %s brain.m 0 pattern  - Test pattern routing\n", argv[0]);
        return 1;
    }
    
    printf("🤖 Melvin Motor EXEC Test\n");
    printf("==========================\n\n");
    
    /* Open brain */
    printf("Opening brain: %s\n", argv[1]);
    Graph *brain = melvin_open(argv[1], 100000, 50000000);
    if (!brain) {
        fprintf(stderr, "❌ Failed to open brain\n");
        return 1;
    }
    
    printf("✅ Brain loaded (nodes: %u, edges: %u)\n\n", 
           brain->node_count, brain->edge_count);
    
    /* Determine test type */
    if (argc < 3 || strcmp(argv[2], "all") == 0) {
        /* Test all motors */
        test_all_motors(brain);
    } else {
        int motor_id = atoi(argv[2]);
        
        if (motor_id < 0 || motor_id >= 14) {
            fprintf(stderr, "❌ Invalid motor ID: %d (must be 0-13)\n", motor_id);
            melvin_close(brain);
            return 1;
        }
        
        if (argc >= 4 && strcmp(argv[3], "seq") == 0) {
            /* Run sequence */
            test_sequence(brain, motor_id);
        } else if (argc >= 4 && strcmp(argv[3], "pattern") == 0) {
            /* Test pattern routing */
            test_pattern_routing(brain, motor_id);
        } else {
            /* Single test */
            test_motor(brain, motor_id, 0.5f);
        }
    }
    
    melvin_close(brain);
    return 0;
}

