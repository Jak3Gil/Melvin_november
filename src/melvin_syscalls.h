/*
 * melvin_syscalls.h - External Syscall Interface
 * 
 * This defines the interface between melvin substrate and host environment.
 * The graph can call these functions to interact with the outside world.
 */

#ifndef MELVIN_SYSCALLS_H
#define MELVIN_SYSCALLS_H

#include <stdint.h>
#include <stddef.h>

/* Forward declare Graph for syscall signatures */
typedef struct Graph Graph;

/* Syscall table - provided by host environment */
typedef struct {
    /* Compilation syscalls - teach graph to compile */
    int (*compile_c_function)(const char *source_code, 
                             const char *function_name,
                             uint8_t *machine_code_out, 
                             size_t *code_size_out);
    
    /* File I/O syscalls - read/write files */
    int (*read_file)(const char *path, uint8_t *buffer, size_t *size);
    int (*write_file)(const char *path, const uint8_t *buffer, size_t size);
    
    /* Tool invocation - STT, TTS, LLM, etc. */
    int (*invoke_tool)(const char *tool_name, 
                      const uint8_t *input, size_t input_size,
                      uint8_t *output, size_t *output_size);
    
    /* Motor control - CAN bus, etc. */
    int (*send_motor_command)(uint32_t motor_id, const uint8_t *frame, size_t frame_size);
    int (*read_motor_feedback)(uint32_t motor_id, uint8_t *frame, size_t *frame_size);
    
    /* Feedback - tell graph if something was right/wrong */
    void (*log_success)(const char *message);
    void (*log_failure)(const char *message);
    
    /* Graph introspection - for debugging */
    void (*print_debug)(const char *message);
    
} MelvinSyscalls;

/* Set syscalls for a graph (called by host) */
void melvin_set_syscalls(Graph *g, MelvinSyscalls *syscalls);

/* Get syscalls from blob (called by EXEC nodes) */
MelvinSyscalls* melvin_get_syscalls_from_blob(Graph *g);

#endif /* MELVIN_SYSCALLS_H */

