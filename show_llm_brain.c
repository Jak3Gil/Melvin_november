/* Show what LLM knowledge looks like in the brain */
#include "src/melvin.h"
#include <stdio.h>

int main() {
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  LLM-ENHANCED BRAIN CONTENTS                          ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    Graph *brain = melvin_open("llm_brain.m", 10000, 50000, 131072);
    if (!brain) {
        printf("❌ Can't open brain\n");
        return 1;
    }
    
    printf("Brain Statistics:\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("  Nodes: %llu\n", (unsigned long long)brain->node_count);
    printf("  Edges: %llu\n", (unsigned long long)brain->edge_count);
    printf("\n");
    
    /* Count and show patterns */
    printf("Patterns Created from LLM Knowledge:\n");
    printf("════════════════════════════════════════════════════════\n");
    
    int pattern_count = 0;
    for (uint64_t i = 840; i < brain->node_count && i < 3000; i++) {
        if (brain->nodes[i].pattern_data_offset > 0) {
            pattern_count++;
            
            if (pattern_count <= 20) {  /* Show first 20 */
                printf("  Pattern %4llu: activation=%.3f, offset=%llu\n",
                       (unsigned long long)i,
                       brain->nodes[i].a,
                       (unsigned long long)brain->nodes[i].pattern_data_offset);
            }
        }
    }
    
    printf("\nTotal patterns: %d\n", pattern_count);
    printf("\n");
    
    /* Show character nodes that were activated by LLM text */
    printf("Character Nodes (from LLM text):\n");
    printf("════════════════════════════════════════════════════════\n");
    
    int active_chars = 0;
    for (uint64_t i = 0; i < 256; i++) {
        if (brain->nodes[i].a > 0.01f) {
            printf("  Node %3llu ('%c'): activation=%.3f\n",
                   (unsigned long long)i,
                   (i >= 32 && i < 127) ? (char)i : '?',
                   brain->nodes[i].a);
            active_chars++;
            if (active_chars >= 30) break;
        }
    }
    printf("\n");
    
    melvin_close(brain);
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  This brain now contains Llama 3 knowledge!          ║\n");
    printf("║  Patterns from: robot rules, conditionals, actions   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    return 0;
}

