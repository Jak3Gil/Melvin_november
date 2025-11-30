/*
 * melvin_run.c - Minimal host runner (NOT a test, just the host loop)
 * 
 * This is what the HOST does:
 *   1. Open .m brain
 *   2. Feed bytes from real sources (stdin, camera, etc.)
 *   3. Call blob entry
 *   4. Repeat
 * 
 * The brain (.m blob) does EVERYTHING else.
 * This file is just the minimal host loop - could be Python, C, whatever.
 */

#include "melvin.h"
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
    const char *brain_path = (argc > 1) ? argv[1] : "brain.m";
    
    /* Open brain */
    Graph *g = melvin_open(brain_path, 1000, 10000, 65536);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", brain_path);
        return 1;
    }
    
    /* Set up syscalls (host provides these) */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    /* Main loop: feed bytes from stdin, let brain run */
    uint8_t buf[4096];
    uint32_t in_port = 256;  /* Port node for input */
    
    while (1) {
        /* Read bytes from stdin (or camera, or whatever) */
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0) break;
        
        /* Feed bytes into brain */
        for (ssize_t i = 0; i < n; i++) {
            melvin_feed_byte(g, in_port, buf[i], 1.0f);
        }
        
        /* Let brain run its laws */
        melvin_call_entry(g);
        
        /* Sync periodically */
        melvin_sync(g);
    }
    
    melvin_close(g);
    return 0;
}

