/*
 * MELVIN I/O IMPLEMENTATION
 * 
 * Implements the I/O interface for Melvin
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "melvin.c"

// Output callback storage
static MelvinOutputCallback g_output_callback = NULL;
static void *g_output_context = NULL;
static MelvinRuntime *g_output_runtime = NULL;

// ========================================================================
// INPUT IMPLEMENTATION
// ========================================================================

void ingest_buffer(MelvinRuntime *rt, uint64_t channel_id, const uint8_t *buffer, size_t len, float energy) {
    if (!rt || !buffer || len == 0) return;
    
    for (size_t i = 0; i < len; i++) {
        ingest_byte(rt, channel_id, buffer[i], energy);
    }
}

int ingest_file(MelvinRuntime *rt, uint64_t channel_id, const char *filename, float energy) {
    if (!rt || !filename) return -1;
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("[ingest_file] fopen");
        return -1;
    }
    
    uint8_t buffer[4096];
    size_t bytes_read;
    size_t total_bytes = 0;
    
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        ingest_buffer(rt, channel_id, buffer, bytes_read, energy);
        total_bytes += bytes_read;
    }
    
    if (ferror(fp)) {
        perror("[ingest_file] fread");
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    printf("[ingest_file] Ingested %zu bytes from %s\n", total_bytes, filename);
    return 0;
}

size_t ingest_stdin(MelvinRuntime *rt, uint64_t channel_id, float energy) {
    if (!rt) return 0;
    
    uint8_t buffer[4096];
    ssize_t bytes_read = read(STDIN_FILENO, buffer, sizeof(buffer));
    
    if (bytes_read > 0) {
        ingest_buffer(rt, channel_id, buffer, bytes_read, energy);
        return bytes_read;
    } else if (bytes_read < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        perror("[ingest_stdin] read");
    }
    
    return 0;
}

// ========================================================================
// OUTPUT IMPLEMENTATION
// ========================================================================

void melvin_register_output_callback(MelvinRuntime *rt, MelvinOutputCallback callback, void *context) {
    g_output_runtime = rt;
    g_output_callback = callback;
    g_output_context = context;
    
    // Ensure output node exists
    if (rt && rt->file) {
        melvin_ensure_param_nodes(rt->file);
        
        // Check if output node exists, create if not
        uint64_t output_idx = find_node_index_by_id(rt->file, NODE_ID_OUTPUT);
        if (output_idx == UINT64_MAX) {
            GraphHeaderDisk *gh = rt->file->graph_header;
            if (gh->num_nodes < gh->node_capacity) {
                uint64_t new_idx = gh->num_nodes++;
                NodeDisk *n = &rt->file->nodes[new_idx];
                n->id = NODE_ID_OUTPUT;
                n->state = 0.0f;
                n->prediction = 0.0f;
                n->bias = 0.0f;
                n->flags = 0;  // Not EXECUTABLE, just a regular output node
                n->first_out_edge = UINT64_MAX;
                n->out_degree = 0;
                n->firing_count = 0;
                n->trace = 0.0f;
                n->prediction_error = 0.0f;
                n->stability = 0.0f;
                n->reward = 0.0f;
            }
        }
    }
}

float melvin_get_output(MelvinRuntime *rt, uint64_t node_id) {
    if (!rt || !rt->file) return 0.0f;
    
    uint64_t idx = find_node_index_by_id(rt->file, node_id);
    if (idx == UINT64_MAX) return 0.0f;
    
    return rt->file->nodes[idx].state;
}

uint8_t melvin_get_output_byte(MelvinRuntime *rt, uint64_t node_id) {
    float activation = melvin_get_output(rt, node_id);
    
    // Map activation [-1, 1] to byte [0, 255]
    // Using tanh-like mapping: activation -> [0, 1] -> [0, 255]
    float normalized = (activation + 1.0f) / 2.0f;  // [-1, 1] -> [0, 1]
    normalized = (normalized < 0.0f) ? 0.0f : (normalized > 1.0f) ? 1.0f : normalized;
    
    return (uint8_t)(normalized * 255.0f);
}

// Check output node and trigger callback if needed
void melvin_check_output(MelvinRuntime *rt) {
    if (!rt || !rt->file || !g_output_callback) return;
    
    uint64_t idx = find_node_index_by_id(rt->file, NODE_ID_OUTPUT);
    if (idx == UINT64_MAX) return;
    
    NodeDisk *output_node = &rt->file->nodes[idx];
    float activation = output_node->state;
    
    // Trigger callback if activation is significant
    if (fabsf(activation) > 0.1f) {
        g_output_callback(rt, NODE_ID_OUTPUT, activation, g_output_context);
    }
}

// ========================================================================
// REWARD IMPLEMENTATION
// ========================================================================

void inject_reward_to_nodes(MelvinRuntime *rt, uint64_t *node_ids, size_t count, float reward_value) {
    if (!rt || !node_ids || count == 0) return;
    
    for (size_t i = 0; i < count; i++) {
        inject_reward(rt, node_ids[i], reward_value);
    }
}

// ========================================================================
// CHANNEL MANAGEMENT
// ========================================================================

uint64_t melvin_get_channel_node_id(uint64_t channel_id) {
    // Channel nodes have IDs: 2000000 + channel_id
    return 2000000ULL + (channel_id & 0xFF);
}

// ========================================================================
// EXEC OUTPUT HELPERS
// ========================================================================

/*
 * Example: How EXEC nodes can output
 * 
 * EXEC nodes run machine code that can:
 * 
 * 1. Write to stdout:
 *    - ARM64: mov x0, #1; adr x1, msg; mov x2, #len; mov x8, #64; svc #0
 *    - x86_64: mov rax, 1; mov rdi, 1; mov rsi, msg; mov rdx, len; syscall
 * 
 * 2. Write to files:
 *    - open() syscall to get file descriptor
 *    - write() syscall to write data
 * 
 * 3. Network I/O:
 *    - socket() syscall to create socket
 *    - connect()/send() to send data
 * 
 * 4. GPU:
 *    - ioctl() to GPU device
 *    - mmap() GPU memory
 *    - Launch kernels via device-specific syscalls
 * 
 * 5. Hardware control:
 *    - open() GPIO/I2C/SPI device files
 *    - ioctl() or write() to control hardware
 * 
 * All of this is done from machine code in EXEC nodes.
 * The return value becomes energy injected back into the graph.
 */

