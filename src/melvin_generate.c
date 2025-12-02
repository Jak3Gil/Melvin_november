/*
 * Melvin Text Generation
 * 
 * Generate text by following wave propagation through the graph:
 * 1. Feed prompt (activates nodes)
 * 2. Run propagation (uel_main)
 * 3. Sample most activated byte nodes
 * 4. Output and repeat (autoregressive)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "melvin.h"

/* Find top-K most activated byte nodes */
void find_top_k_bytes(Graph *g, int k, uint32_t *out_nodes, float *out_activations) {
    /* Check byte nodes (0-255) for activation */
    for (int i = 0; i < k; i++) {
        out_nodes[i] = UINT32_MAX;
        out_activations[i] = -INFINITY;
    }
    
    for (uint32_t node_id = 0; node_id < 256 && node_id < g->node_count; node_id++) {
        float activation = fabsf(g->nodes[node_id].a);
        
        /* Insert into top-K if high enough */
        for (int i = 0; i < k; i++) {
            if (activation > out_activations[i]) {
                /* Shift down */
                for (int j = k-1; j > i; j--) {
                    out_nodes[j] = out_nodes[j-1];
                    out_activations[j] = out_activations[j-1];
                }
                out_nodes[i] = node_id;
                out_activations[i] = activation;
                break;
            }
        }
    }
}

/* Temperature-based sampling */
uint32_t sample_with_temperature(uint32_t *nodes, float *activations, int k, float temperature) {
    if (k == 0 || nodes[0] == UINT32_MAX) return UINT32_MAX;
    
    /* Convert activations to probabilities with temperature */
    float probs[k];
    float sum = 0.0f;
    
    for (int i = 0; i < k; i++) {
        if (nodes[i] == UINT32_MAX) break;
        probs[i] = expf(activations[i] / temperature);
        sum += probs[i];
    }
    
    if (sum == 0.0f) return nodes[0];  /* Return most activated */
    
    /* Normalize */
    for (int i = 0; i < k; i++) {
        probs[i] /= sum;
    }
    
    /* Sample */
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    
    for (int i = 0; i < k; i++) {
        if (nodes[i] == UINT32_MAX) break;
        cumsum += probs[i];
        if (r <= cumsum) {
            return nodes[i];
        }
    }
    
    return nodes[0];  /* Fallback */
}

/* Generate text given a prompt */
int melvin_generate(Graph *g, const char *prompt, char *output, int max_output_len, 
                   float temperature, int top_k) {
    if (!g || !prompt || !output || max_output_len <= 0) return -1;
    
    /* Feed prompt to activate initial nodes */
    printf("Prompt: \"%s\"\n", prompt);
    printf("Generating");
    fflush(stdout);
    
    for (int i = 0; prompt[i] != '\0'; i++) {
        melvin_feed_byte(g, 0, (uint8_t)prompt[i], 1.0f);
    }
    
    /* Run propagation to let patterns activate */
    extern void uel_main(Graph *g);  /* Declare if not in header */
    
    /* Generate tokens */
    int output_pos = 0;
    int consecutive_spaces = 0;
    const int MAX_CONSECUTIVE_SPACES = 3;
    
    for (int step = 0; step < max_output_len; step++) {
        /* Run propagation (let energy flow through graph) */
        uel_main(g);
        
        /* Find most activated byte nodes */
        uint32_t top_nodes[10];
        float top_activations[10];
        find_top_k_bytes(g, top_k < 10 ? top_k : 10, top_nodes, top_activations);
        
        /* Sample next byte */
        uint32_t next_node = sample_with_temperature(top_nodes, top_activations, 10, temperature);
        
        if (next_node == UINT32_MAX || next_node >= 256) {
            break;  /* No valid byte found */
        }
        
        uint8_t next_byte = (uint8_t)next_node;
        
        /* Filter output (skip non-printable, limit spaces) */
        if (next_byte < 32 && next_byte != '\n') {
            continue;  /* Skip control characters */
        }
        
        if (next_byte == ' ' || next_byte == '\n') {
            consecutive_spaces++;
            if (consecutive_spaces >= MAX_CONSECUTIVE_SPACES) {
                continue;  /* Skip excessive spaces */
            }
        } else {
            consecutive_spaces = 0;
        }
        
        /* Output byte */
        output[output_pos++] = next_byte;
        
        /* Show progress */
        if (step % 10 == 0) {
            printf(".");
            fflush(stdout);
        }
        
        /* Feed back to graph (autoregressive) */
        melvin_feed_byte(g, 0, next_byte, 0.5f);
        
        /* Stop on natural ending (period, question mark) */
        if (next_byte == '.' || next_byte == '?' || next_byte == '!') {
            if (output_pos > 20) {  /* Minimum length */
                break;
            }
        }
    }
    
    output[output_pos] = '\0';
    printf("\n");
    
    return output_pos;
}

/* Test generation */
int main(int argc, char *argv[]) {
    printf("==============================================\n");
    printf("MELVIN TEXT GENERATION TEST\n");
    printf("==============================================\n\n");
    
    /* Load pre-trained brain or create new */
    const char *brain_path = (argc > 1) ? argv[1] : "/tmp/generation_test.m";
    Graph *g = melvin_open(brain_path, 5000, 25000, 8192);
    
    if (!g) {
        printf("Creating new brain and training on sample text...\n");
        
        /* Create brain */
        if (melvin_create_v2(brain_path, 5000, 25000, 8192, 0) != 0) {
            fprintf(stderr, "Failed to create brain\n");
            return 1;
        }
        
        g = melvin_open(brain_path, 5000, 25000, 8192);
        if (!g) {
            fprintf(stderr, "Failed to open brain\n");
            return 1;
        }
        
        /* Train on sample text */
        const char *training_text = 
            "To be or not to be, that is the question. "
            "Whether tis nobler in the mind to suffer. "
            "The slings and arrows of outrageous fortune. "
            "To be or not to be. To be. To be or not.";
        
        printf("Training on: \"%s\"\n\n", training_text);
        
        for (int i = 0; training_text[i] != '\0'; i++) {
            melvin_feed_byte(g, 0, (uint8_t)training_text[i], 1.0f);
        }
    }
    
    printf("Brain loaded: %llu nodes, %llu edges\n\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    
    /* Test prompts */
    const char *prompts[] = {
        "To be or ",
        "To be or not to ",
        "Whether ",
        NULL
    };
    
    for (int i = 0; prompts[i] != NULL; i++) {
        char output[200];
        
        printf("--- Test %d ---\n", i+1);
        int len = melvin_generate(g, prompts[i], output, 100, 
                                   1.0f,   /* temperature */
                                   5);     /* top_k */
        
        if (len > 0) {
            printf("Output: \"%s\"\n", output);
            printf("Length: %d characters\n", len);
        } else {
            printf("Generation failed\n");
        }
        printf("\n");
    }
    
    melvin_close(g);
    
    printf("==============================================\n");
    printf("Generation test complete!\n");
    printf("==============================================\n");
    
    return 0;
}

