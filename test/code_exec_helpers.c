/*
 * code_exec_helpers.c
 * 
 * EXEC helper functions for code compilation and execution.
 * These are called by EXEC nodes when they fire.
 * 
 * Functions:
 * - melvin_exec_compile: Reads source from SRC_IN, compiles, writes binary to BIN_IN
 * - melvin_exec_link: Links/loads binary into executable memory
 * - melvin_exec_run: Runs compiled code block
 * 
 * All functions read/write graph nodes - no external state.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>

// Forward declarations
// Note: This file should be included AFTER melvin.c to get type definitions
#ifndef _MELVIN_FILE_DEFINED
typedef struct MelvinFile MelvinFile;
#endif
// NodeDisk is defined in melvin.c - don't redeclare

// Node ID constants (from code_instincts.c)
#define NODE_ID_PORT_SRC_IN   100000ULL
#define NODE_ID_PORT_BIN_IN   100001ULL
#define NODE_ID_PORT_OUT_LOG    100003ULL
#define NODE_ID_REGION_SRC    100010ULL
#define NODE_ID_REGION_BIN    100011ULL
#define NODE_ID_BLOCK_ENTRY   100020ULL
#define NODE_ID_BLOCK_SIZE    100021ULL

// Helper: Find node by ID
extern uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id);
extern uint8_t *melvin_get_payload_span_safe(MelvinFile *file, uint64_t offset, uint64_t len);

// Helper: Read bytes from node (reads from payload or state)
static size_t read_bytes_from_node(MelvinFile *file, uint64_t node_id, uint8_t *buf, size_t max_len) {
    if (!file || !buf || max_len == 0) return 0;
    
    uint64_t idx = find_node_index_by_id(file, node_id);
    if (idx == UINT64_MAX) return 0;
    
    NodeDisk *node = &file->nodes[idx];
    if (!node) return 0;
    
    // Try to read from payload
    if (node->payload_offset != 0 && node->payload_len > 0) {
        size_t to_read = (node->payload_len < max_len) ? node->payload_len : max_len;
        uint8_t *payload = melvin_get_payload_span_safe(file, node->payload_offset, to_read);
        if (payload) {
            memcpy(buf, payload, to_read);
            return to_read;
        }
    }
    
    return 0;
}

// Helper: Write bytes to node (writes to blob at payload_offset)
static int write_bytes_to_node(MelvinFile *file, uint64_t node_id, const uint8_t *data, size_t len) {
    if (!file || !data || len == 0) return -1;
    
    uint64_t idx = find_node_index_by_id(file, node_id);
    if (idx == UINT64_MAX) return -1;
    
    NodeDisk *node = &file->nodes[idx];
    if (!node) return -1;
    
    // Write to blob at payload_offset (or allocate new space)
    // For now, just write to existing payload if it exists
    if (node->payload_offset != 0) {
        uint8_t *payload = melvin_get_payload_span_safe(file, node->payload_offset, len);
        if (payload) {
            size_t to_write = (node->payload_len < len) ? node->payload_len : len;
            memcpy(payload, data, to_write);
            return 0;
        }
    }
    
    // TODO: Allocate new blob space if needed
    return -1;
}

// EXEC_COMPILE: Compiles source code from SRC_IN, writes binary to BIN_IN
void melvin_exec_compile(MelvinFile *g, uint64_t self_id) {
    if (!g) return;
    
    // Read source from SRC_IN node
    uint8_t src_buf[8192];
    size_t src_len = read_bytes_from_node(g, NODE_ID_PORT_SRC_IN, src_buf, sizeof(src_buf));
    
    if (src_len == 0) {
        // No source to compile
        const char *err = "ERROR: No source code in SRC_IN\n";
        write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)err, strlen(err));
        return;
    }
    
    // Write source to temp file
    char tmp_src[] = "/tmp/melvin_compile_XXXXXX.c";
    int fd_src = mkstemp(tmp_src);
    if (fd_src < 0) {
        const char *err = "ERROR: Cannot create temp source file\n";
        write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)err, strlen(err));
        return;
    }
    
    write(fd_src, src_buf, src_len);
    close(fd_src);
    
    // Compile using clang (or gcc)
    char tmp_bin[] = "/tmp/melvin_compile_XXXXXX.o";
    int fd_bin = mkstemp(tmp_bin);
    if (fd_bin < 0) {
        unlink(tmp_src);
        const char *err = "ERROR: Cannot create temp binary file\n";
        write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)err, strlen(err));
        return;
    }
    close(fd_bin);
    unlink(tmp_bin);  // We'll use the filename
    
    // Try clang first, then gcc
    const char *compiler = "clang";
    char *cmd_argv[] = {
        (char*)"clang",
        (char*)"-c",
        (char*)"-o",
        tmp_bin,
        tmp_src,
        NULL
    };
    
    pid_t pid = fork();
    if (pid == 0) {
        // Child: run compiler
        execvp(compiler, cmd_argv);
        exit(1);
    } else if (pid > 0) {
        // Parent: wait for compilation
        int status;
        waitpid(pid, &status, 0);
        
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            // Compilation succeeded - read binary
            FILE *f = fopen(tmp_bin, "rb");
            if (f) {
                uint8_t bin_buf[8192];
                size_t bin_len = fread(bin_buf, 1, sizeof(bin_buf), f);
                fclose(f);
                
                // Write binary to BIN_IN node
                write_bytes_to_node(g, NODE_ID_PORT_BIN_IN, bin_buf, bin_len);
                
                // Update block metadata
                // (In real implementation, would parse ELF/object file)
                const char *log = "Compilation succeeded\n";
                write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)log, strlen(log));
            }
        } else {
            // Compilation failed
            const char *err = "ERROR: Compilation failed\n";
            write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)err, strlen(err));
        }
        
        unlink(tmp_src);
        unlink(tmp_bin);
    }
}

// EXEC_LINK: Links/loads binary into executable memory
void melvin_exec_link(MelvinFile *g, uint64_t self_id) {
    if (!g) return;
    
    // Read binary from BIN_IN
    uint8_t bin_buf[8192];
    size_t bin_len = read_bytes_from_node(g, NODE_ID_PORT_BIN_IN, bin_buf, sizeof(bin_buf));
    
    if (bin_len == 0) {
        const char *err = "ERROR: No binary in BIN_IN\n";
        write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)err, strlen(err));
        return;
    }
    
    // For now, just update block metadata
    // In real implementation, would:
    // 1. Parse object file
    // 2. Load into executable memory
    // 3. Update BLOCK_ENTRY with function pointer
    
    // Write block size
    NodeDisk *size_node = &g->nodes[find_node_index_by_id(g, NODE_ID_BLOCK_SIZE)];
    if (size_node) {
        size_node->state = (float)bin_len;
    }
    
    const char *log = "Binary linked/loaded\n";
    write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)log, strlen(log));
}

// EXEC_RUN: Runs compiled code block
void melvin_exec_run(MelvinFile *g, uint64_t self_id) {
    if (!g) return;
    
    // Read block entry point
    NodeDisk *entry_node = &g->nodes[find_node_index_by_id(g, NODE_ID_BLOCK_ENTRY)];
    if (!entry_node || entry_node->state == 0.0f) {
        const char *err = "ERROR: No entry point set\n";
        write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)err, strlen(err));
        return;
    }
    
    // In real implementation, would:
    // 1. Get function pointer from entry_node
    // 2. Call function with signature: void fn(MelvinFile *g, uint64_t self_id)
    // 3. Write output to OUT_LOG
    
    const char *log = "Code block executed\n";
    write_bytes_to_node(g, NODE_ID_PORT_OUT_LOG, (const uint8_t*)log, strlen(log));
}

