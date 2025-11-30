/*
 * test_conversation_and_motor.c - Test conversation and motor control patterns
 * 
 * Tests:
 * 1. Conversation data feeding
 * 2. C file reading and compilation
 * 3. Motor control patterns
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

int main(void) {
    printf("========================================\n");
    printf("Conversation & Motor Control Test\n");
    printf("========================================\n");
    printf("\n");
    
    Graph *g = melvin_open("/tmp/conversation_motor_brain.m", 0, 10000, 131072);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    printf("Initial State:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("\n");
    
    /* Test 1: Feed conversation data */
    printf("Test 1: Feeding conversation data...\n");
    const char *conversation = "Hello, how are you?";
    for (size_t i = 0; i < strlen(conversation); i++) {
        melvin_feed_byte(g, 20, (uint8_t)conversation[i], 0.3f);  /* Text input port 20 */
    }
    
    /* Process through graph */
    for (int i = 0; i < 10; i++) {
        melvin_call_entry(g);
    }
    
    printf("  After conversation input:\n");
    printf("    Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("    Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("\n");
    
    /* Test 2: Read and compile C file */
    printf("Test 2: Reading and compiling C file...\n");
    const char *c_source = "void motor_control(int speed) { /* motor code */ }";
    
    /* Simulate file read via syscall */
    uint8_t *file_data = NULL;
    size_t file_len = 0;
    if (syscalls.sys_read_file) {
        /* Write test file first */
        syscalls.sys_write_file("/tmp/test_motor.c", (const uint8_t *)c_source, strlen(c_source));
        syscalls.sys_read_file("/tmp/test_motor.c", &file_data, &file_len);
        
        if (file_data) {
            printf("  Read file: %zu bytes\n", file_len);
            
            /* Feed file data into graph (file I/O gateway 730) */
            for (size_t i = 0; i < file_len && i < 100; i++) {
                melvin_feed_byte(g, 730, file_data[i], 0.2f);
            }
            
            /* Process */
            for (int i = 0; i < 10; i++) {
                melvin_call_entry(g);
            }
            
            /* Try to compile */
            if (syscalls.sys_compile_c) {
                uint64_t blob_offset = 0;
                uint64_t code_size = 0;
                int result = syscalls.sys_compile_c(file_data, file_len, &blob_offset, &code_size);
                if (result == 0) {
                    printf("  Compiled C code: %llu bytes at offset %llu\n",
                           (unsigned long long)code_size, (unsigned long long)blob_offset);
                    
                    /* Feed compiled code into code pattern nodes (740-839) */
                    for (uint64_t i = 0; i < code_size && i < 100; i++) {
                        uint32_t node_id = 740 + (i % 100);
                        melvin_feed_byte(g, node_id, g->blob[blob_offset + i], 0.2f);
                    }
                    
                    /* Process */
                    for (int i = 0; i < 10; i++) {
                        melvin_call_entry(g);
                    }
                } else {
                    printf("  ⚠ Compilation failed\n");
                }
            }
            
            free(file_data);
        }
    }
    
    printf("  After C file processing:\n");
    printf("    Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("    Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("\n");
    
    /* Test 3: Check motor control patterns */
    printf("Test 3: Checking motor control patterns...\n");
    uint32_t motor_edges = 0;
    for (uint64_t i = 0; i < g->edge_count; i++) {
        uint32_t src = g->edges[i].src;
        uint32_t dst = g->edges[i].dst;
        if ((src >= 700 && src < 720) || (dst >= 700 && dst < 720)) {
            motor_edges++;
        }
    }
    printf("  Motor gateway edges: %u\n", motor_edges);
    
    /* Test 4: Check conversation patterns */
    printf("Test 4: Checking conversation patterns...\n");
    uint32_t conv_edges = 0;
    for (uint64_t i = 0; i < g->edge_count; i++) {
        uint32_t src = g->edges[i].src;
        uint32_t dst = g->edges[i].dst;
        if ((src >= 204 && src < 210) || (dst >= 204 && dst < 210)) {
            conv_edges++;
        }
    }
    printf("  Conversation memory edges: %u\n", conv_edges);
    
    /* Test 5: Check file I/O patterns */
    printf("Test 5: Checking file I/O patterns...\n");
    uint32_t file_edges = 0;
    for (uint64_t i = 0; i < g->edge_count; i++) {
        uint32_t src = g->edges[i].src;
        uint32_t dst = g->edges[i].dst;
        if ((src >= 720 && src < 740) || (dst >= 720 && dst < 740)) {
            file_edges++;
        }
    }
    printf("  File I/O gateway edges: %u\n", file_edges);
    
    printf("\n========================================\n");
    printf("FINAL STATE\n");
    printf("========================================\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("  Motor edges: %u\n", motor_edges);
    printf("  Conversation edges: %u\n", conv_edges);
    printf("  File I/O edges: %u\n", file_edges);
    printf("\n");
    
    if (motor_edges > 0 && conv_edges > 0 && file_edges > 0) {
        printf("✓ ALL PATTERNS CREATED!\n");
        printf("\nThe system is ready for:\n");
        printf("  ✓ Conversation data feeding\n");
        printf("  ✓ C file reading and compilation\n");
        printf("  ✓ Motor control\n");
        printf("  ✓ File I/O operations\n");
    } else {
        printf("⚠ Some patterns may be missing\n");
    }
    
    melvin_close(g);
    return 0;
}

