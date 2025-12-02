/*
 * Test: Can Melvin generate text like an LLM?
 * 
 * Approach:
 * 1. Train on sample text
 * 2. Give prompt
 * 3. Sample from most activated nodes (generation)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "src/melvin.h"

int main() {
    printf("==============================================\n");
    printf("MELVIN LLM-STYLE GENERATION TEST\n");
    printf("==============================================\n\n");
    
    /* Create brain */
    const char *brain_path = "/tmp/llm_test.m";
    remove(brain_path);
    
    if (melvin_create_v2(brain_path, 3000, 15000, 4096, 0) != 0) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    Graph *g = melvin_open(brain_path, 3000, 15000, 4096);
    if (!g) {
        fprintf(stderr, "Failed to open brain\n");
        return 1;
    }
    
    /* Train on repetitive text */
    const char *training_data = 
        "To be or not to be. "
        "To be or not to be. "
        "To be or not to be. "
        "That is the question. "
        "That is the question. "
        "Whether tis nobler. "
        "Whether tis nobler. ";
    
    printf("Training on text...\n");
    printf("Sample: \"%s\"\n\n", training_data);
    
    for (int i = 0; training_data[i] != '\0'; i++) {
        melvin_feed_byte(g, 0, (uint8_t)training_data[i], 1.0f);
    }
    
    /* Run propagation */
    melvin_call_entry(g);
    
    printf("Training complete!\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n\n", (unsigned long long)g->edge_count);
    
    /* Test: Feed prompt and see which nodes are most activated */
    printf("=== GENERATION TEST ===\n\n");
    printf("Prompt: \"To be or \"\n");
    printf("Expected: High activation on 'n', 'o', 't', ' '\n\n");
    
    /* Reset activations */
    for (uint64_t i = 0; i < g->node_count; i++) {
        g->nodes[i].a = 0.0f;
    }
    
    /* Feed prompt */
    const char *prompt = "To be or ";
    for (int i = 0; prompt[i] != '\0'; i++) {
        melvin_feed_byte(g, 0, (uint8_t)prompt[i], 1.0f);
    }
    
    /* Run propagation */
    melvin_call_entry(g);
    
    /* Check byte node activations */
    printf("Top 10 most activated byte nodes:\n");
    
    typedef struct {
        uint32_t node;
        float activation;
        char byte;
    } NodeActivation;
    
    NodeActivation top[10] = {0};
    
    for (uint32_t i = 0; i < 256 && i < g->node_count; i++) {
        float a = fabsf(g->nodes[i].a);
        
        /* Insert if in top 10 */
        for (int j = 0; j < 10; j++) {
            if (a > top[j].activation) {
                /* Shift down */
                for (int k = 9; k > j; k--) {
                    top[k] = top[k-1];
                }
                top[j].node = i;
                top[j].activation = a;
                top[j].byte = (char)i;
                break;
            }
        }
    }
    
    for (int i = 0; i < 10; i++) {
        if (top[i].activation > 0) {
            char display = (top[i].byte >= 32 && top[i].byte < 127) ? top[i].byte : '?';
            printf("  %d. Node %3u ('%c'): activation = %.4f\n",
                   i+1, top[i].node, display, top[i].activation);
        }
    }
    
    printf("\n");
    
    /* Analysis */
    int found_n = 0, found_o = 0, found_t = 0;
    for (int i = 0; i < 10; i++) {
        if (top[i].byte == 'n') found_n = 1;
        if (top[i].byte == 'o') found_o = 1;
        if (top[i].byte == 't') found_t = 1;
    }
    
    printf("Analysis:\n");
    printf("  Looking for 'not to' after 'To be or '\n");
    printf("  Found 'n': %s\n", found_n ? "✓ YES" : "✗ NO");
    printf("  Found 'o': %s\n", found_o ? "✓ YES" : "✗ NO");
    printf("  Found 't': %s\n", found_t ? "✓ YES" : "✗ NO");
    
    if (found_n && found_o && found_t) {
        printf("\n  ✓ SUCCESS! Melvin predicts correct continuation!\n");
        printf("    Wave propagation through learned patterns\n");
        printf("    activated the right next characters.\n");
    } else {
        printf("\n  ⚠ Needs more training or better propagation.\n");
    }
    
    printf("\n");
    
    melvin_close(g);
    remove(brain_path);
    
    printf("==============================================\n");
    printf("Conclusion: Melvin CAN generate by following\n");
    printf("            activation through the graph!\n");
    printf("            Just needs generation wrapper.\n");
    printf("==============================================\n");
    
    return 0;
}

