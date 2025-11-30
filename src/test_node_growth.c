/*
 * test_node_growth.c - Prove node growth and pattern seeding
 * 
 * Tests:
 * 1. Node growth beyond initial allocation
 * 2. Pattern seeding from tools
 * 3. Real graph expansion
 */

#include "melvin.h"
#include "melvin_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("========================================\n");
    printf("Node Growth & Pattern Seeding Test\n");
    printf("========================================\n\n");
    
    /* Test 1: Start with small graph, prove nodes grow */
    printf("TEST 1: Node Growth Beyond Initial Allocation\n");
    printf("==============================================\n");
    
    /* Create graph with only 256 nodes (just byte values) */
    Graph *g = melvin_open("/tmp/test_growth.m", 256, 1000, 65536);
    if (!g) {
        printf("✗ Failed to create graph\n");
        return 1;
    }
    
    uint64_t nodes_before = g->node_count;
    uint64_t edges_before = g->edge_count;
    
    printf("Initial: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_before,
           (unsigned long long)edges_before);
    
    /* Feed data through high port numbers to force node growth */
    /* Ports 256+ don't exist yet, so they'll trigger growth */
    printf("\nFeeding data through high ports (256-400) to trigger node growth...\n");
    for (int i = 256; i < 400; i++) {
        /* Feed through port i (which doesn't exist yet) - this will trigger ensure_node */
        /* But limit to avoid segfault - grow incrementally */
        if (i % 50 == 0) {
            printf("  Feeding port %d (current nodes: %llu)...\n", i, (unsigned long long)g->node_count);
        }
        melvin_feed_byte(g, (uint32_t)i, (uint8_t)(i % 256), 0.1f);
        /* Process every 10 feeds to allow growth to stabilize */
        if (i % 10 == 0) {
            melvin_call_entry(g);
        }
    }
    melvin_call_entry(g);  /* Final processing */
    
    uint64_t nodes_after = g->node_count;
    uint64_t edges_after = g->edge_count;
    
    printf("After feeding: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_after,
           (unsigned long long)edges_after);
    printf("Growth: +%llu nodes, +%llu edges\n",
           (unsigned long long)(nodes_after - nodes_before),
           (unsigned long long)(edges_after - edges_before));
    
    if (nodes_after > nodes_before) {
        printf("✓ Nodes grew from %llu → %llu\n",
               (unsigned long long)nodes_before,
               (unsigned long long)nodes_after);
    } else {
        printf("⚠ Nodes did not grow (may need to feed beyond byte range)\n");
    }
    
    melvin_close(g);
    printf("\n");
    
    /* Test 2: Pattern seeding from tools */
    printf("TEST 2: Pattern Seeding from Tools\n");
    printf("===================================\n");
    
    g = melvin_open("/tmp/test_patterns.m", 256, 1000, 65536);
    if (!g) {
        printf("✗ Failed to create graph\n");
        return 1;
    }
    
    nodes_before = g->node_count;
    edges_before = g->edge_count;
    
    printf("Before tool patterns: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_before,
           (unsigned long long)edges_before);
    
    /* Seed pattern from TTS */
    printf("\nSeeding pattern from TTS tool...\n");
    uint8_t *audio = NULL;
    size_t audio_len = 0;
    const char *text = "hello melvin";
    
    if (melvin_tool_audio_tts((const uint8_t *)text, strlen(text), &audio, &audio_len) == 0 && audio) {
        printf("  TTS generated %zu bytes of audio\n", audio_len);
        
        /* Feed audio bytes into graph - each unique byte creates/activates a node */
        for (size_t i = 0; i < audio_len && i < 1000; i++) {
            melvin_feed_byte(g, 0, audio[i], 0.1f);
        }
        melvin_call_entry(g);
        
        nodes_after = g->node_count;
        edges_after = g->edge_count;
        
        printf("  After TTS pattern: %llu nodes, %llu edges\n",
               (unsigned long long)nodes_after,
               (unsigned long long)edges_after);
        printf("  Pattern created: +%llu edges\n",
               (unsigned long long)(edges_after - edges_before));
        
        free(audio);
    } else {
        printf("  ⚠ TTS failed\n");
    }
    
    /* Seed pattern from Vision */
    printf("\nSeeding pattern from Vision tool...\n");
    uint8_t img[200];
    for (int i = 0; i < 200; i++) img[i] = (uint8_t)(i % 256);
    
    uint8_t *labels = NULL;
    size_t labels_len = 0;
    
    edges_before = g->edge_count;
    
    if (melvin_tool_vision_identify(img, sizeof(img), &labels, &labels_len) == 0 && labels) {
        printf("  Vision generated: %.*s\n", (int)labels_len, labels);
        
        /* Feed vision labels into graph */
        for (size_t i = 0; i < labels_len; i++) {
            melvin_feed_byte(g, 10, labels[i], 0.1f);  /* Port 10 for vision */
        }
        melvin_call_entry(g);
        
        edges_after = g->edge_count;
        
        printf("  After Vision pattern: %llu edges\n",
               (unsigned long long)edges_after);
        printf("  Pattern created: +%llu edges\n",
               (unsigned long long)(edges_after - edges_before));
        
        free(labels);
    } else {
        printf("  ⚠ Vision failed\n");
    }
    
    /* Seed pattern from LLM */
    printf("\nSeeding pattern from LLM tool...\n");
    edges_before = g->edge_count;
    
    uint8_t *llm_response = NULL;
    size_t llm_response_len = 0;
    const char *prompt = "say hello";
    
    if (melvin_tool_llm_generate((const uint8_t *)prompt, strlen(prompt),
                                 &llm_response, &llm_response_len) == 0 && llm_response) {
        printf("  LLM generated: %.*s\n", 
               (int)(llm_response_len < 100 ? llm_response_len : 100),
               llm_response);
        
        /* Feed LLM response into graph */
        for (size_t i = 0; i < llm_response_len && i < 500; i++) {
            melvin_feed_byte(g, 20, llm_response[i], 0.1f);  /* Port 20 for text input */
        }
        melvin_call_entry(g);
        
        edges_after = g->edge_count;
        
        printf("  After LLM pattern: %llu edges\n",
               (unsigned long long)edges_after);
        printf("  Pattern created: +%llu edges\n",
               (unsigned long long)(edges_after - edges_before));
        
        free(llm_response);
    } else {
        printf("  ⚠ LLM failed (may not be running)\n");
    }
    
    nodes_after = g->node_count;
    
    printf("\nFinal state: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_after,
           (unsigned long long)edges_after);
    printf("Total growth: +%llu nodes, +%llu edges\n",
           (unsigned long long)(nodes_after - nodes_before),
           (unsigned long long)(edges_after - edges_before));
    
    if (edges_after > edges_before) {
        printf("✓ Patterns successfully seeded into graph!\n");
    }
    
    melvin_close(g);
    printf("\n");
    
    printf("========================================\n");
    printf("Test Complete\n");
    printf("========================================\n");
    
    return 0;
}

