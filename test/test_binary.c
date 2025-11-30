/* Test pure loader - no physics in runtime */
#include "melvin.h"
#include <stdio.h>

int main(void) {
    /* Open brain */
    Graph *g = melvin_open("test_brain.m", 1000, 10000, 65536);
    if (!g) {
        printf("FAIL: Could not open brain.m\n");
        return 1;
    }
    
    printf("[OK] Opened brain.m\n");
    printf("     Nodes: %llu, Edges: %llu\n", 
           (unsigned long long)g->hdr->node_count,
           (unsigned long long)g->hdr->edge_count);
    printf("     Main entry: %llu\n",
           (unsigned long long)g->hdr->main_entry_offset);
    
    /* Feed byte (just writes to .m, no physics) */
    melvin_feed_byte(g, 256, 'A', 1.0f);
    printf("[OK] Fed 'A' through node 256\n");
    
    float a_before = melvin_get_activation(g, 'A');
    printf("     Node 'A' activation before: %.4f\n", a_before);
    
    /* Set up syscalls */
    MelvinSyscalls syscalls = {0};
    syscalls.sys_write_text = (void (*)(const uint8_t *, size_t))printf;
    melvin_set_syscalls(g, &syscalls);
    printf("[OK] Set syscalls\n");
    
    /* Call entry (jump into blob - blob does ALL physics) */
    if (g->hdr->main_entry_offset != 0) {
        printf("[OK] Calling blob entrypoint...\n");
        melvin_call_entry(g);
        printf("[OK] Returned from blob\n");
        
        float a_after = melvin_get_activation(g, 'A');
        printf("     Node 'A' activation after: %.4f\n", a_after);
    } else {
        printf("[INFO] No entrypoint set (blob empty)\n");
        printf("       Run uel_seed_tool to seed blob with UEL physics\n");
    }
    
    melvin_sync(g);
    melvin_close(g);
    
    printf("[OK] Closed brain.m\n");
    return 0;
}
