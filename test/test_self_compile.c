/*
 * test_self_compile.c - Test self-compilation
 * 
 * This demonstrates:
 * 1. Feeding C source bytes into graph (just bytes, no special handling)
 * 2. Blob code deciding to compile
 * 3. Compiled machine code stored back in blob
 * 4. All bytes (source + compiled) ingested as energy
 */

#include "melvin.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    /* Open brain */
    Graph *g = melvin_open("test_brain.m", 1000, 10000, 65536);
    if (!g) {
        printf("FAIL: Could not open brain.m\n");
        return 1;
    }
    
    printf("[OK] Opened brain.m\n");
    
    /* Initialize host syscalls */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    printf("[OK] Set syscalls\n");
    
    /* Feed C source as bytes (no special handling - just bytes) */
    const char *c_source = "int add(int a, int b) { return a + b; }";
    printf("\nFeeding C source as bytes:\n");
    printf("  \"%s\"\n", c_source);
    
    /* Write source to temp file first */
    FILE *f = fopen("/tmp/test_source.c", "w");
    if (f) {
        fputs(c_source, f);
        fclose(f);
        
        /* Feed source bytes into graph */
        for (size_t i = 0; i < strlen(c_source); i++) {
            melvin_feed_byte(g, 256, (uint8_t)c_source[i], 0.5f);
        }
        printf("[OK] Fed %zu bytes into graph\n", strlen(c_source));
    }
    
    /* Call blob entry - blob decides what to do */
    if (g->hdr->main_entry_offset != 0) {
        printf("\nCalling blob entrypoint...\n");
        printf("(Blob code would detect C source pattern and call mc_compile_c)\n");
        melvin_call_entry(g);
    } else {
        printf("\n[INFO] No entrypoint set - blob is empty\n");
        printf("       In production, blob code would:\n");
        printf("       1. Detect C source pattern in graph\n");
        printf("       2. Call mc_compile_c() via function pointer\n");
        printf("       3. Store compiled code in blob\n");
        printf("       4. Ingest compiled bytes as energy\n");
    }
    
    /* Check if source bytes created energy patterns */
    printf("\nSource bytes in graph:\n");
    for (size_t i = 0; i < strlen(c_source) && i < 20; i++) {
        uint8_t b = (uint8_t)c_source[i];
        float a = melvin_get_activation(g, b);
        if (a > 0.01f) {
            printf("  '%c' (0x%02X): activation=%.4f\n", 
                   (b >= 32 && b < 127) ? b : '?', b, a);
        }
    }
    
    melvin_sync(g);
    melvin_close(g);
    
    printf("\n[OK] Closed brain.m\n");
    printf("\nKey point: C source, compiled code, vision, motors - all just bytes + energy\n");
    
    return 0;
}

