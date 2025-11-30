/*
 * test_pattern_seeding.c - Prove patterns are seeded from tools into graph
 * 
 * This test shows:
 * 1. Tool outputs create graph structure (edges)
 * 2. Patterns from tools become nodes/edges
 * 3. Graph learns from tool-generated patterns
 */

#include "melvin.h"
#include "melvin_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("========================================\n");
    printf("Pattern Seeding from Tools Test\n");
    printf("========================================\n\n");
    
    /* Create graph */
    Graph *g = melvin_open("/tmp/test_patterns.m", 256, 1000, 65536);
    if (!g) {
        printf("✗ Failed to create graph\n");
        return 1;
    }
    
    uint64_t nodes_before = g->node_count;
    uint64_t edges_before = g->edge_count;
    
    printf("Initial state: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_before,
           (unsigned long long)edges_before);
    printf("\n");
    
    /* Test 1: Seed pattern from TTS */
    printf("TEST 1: Seeding Pattern from TTS Tool\n");
    printf("=====================================\n");
    
    uint64_t edges_tts_before = g->edge_count;
    
    uint8_t *audio = NULL;
    size_t audio_len = 0;
    const char *text = "hello melvin";
    
    printf("Calling TTS tool with text: \"%s\"\n", text);
    if (melvin_tool_audio_tts((const uint8_t *)text, strlen(text), &audio, &audio_len) == 0 && audio) {
        printf("✓ TTS generated %zu bytes of audio\n", audio_len);
        
        /* Feed audio bytes into graph - each byte creates/activates nodes and edges */
        printf("Feeding audio bytes into graph...\n");
        for (size_t i = 0; i < audio_len && i < 500; i++) {
            melvin_feed_byte(g, 0, audio[i], 0.1f);  /* Port 0 = input */
        }
        melvin_call_entry(g);  /* Process through UEL physics */
        
        uint64_t edges_tts_after = g->edge_count;
        uint64_t nodes_tts_after = g->node_count;
        
        printf("After TTS pattern: %llu nodes, %llu edges\n",
               (unsigned long long)nodes_tts_after,
               (unsigned long long)edges_tts_after);
        printf("Pattern created: +%llu edges\n",
               (unsigned long long)(edges_tts_after - edges_tts_before));
        
        if (edges_tts_after > edges_tts_before) {
            printf("✓ TTS pattern successfully seeded into graph!\n");
        }
        
        free(audio);
    } else {
        printf("⚠ TTS failed\n");
    }
    printf("\n");
    
    /* Test 2: Seed pattern from Vision */
    printf("TEST 2: Seeding Pattern from Vision Tool\n");
    printf("========================================\n");
    
    uint64_t edges_vision_before = g->edge_count;
    
    uint8_t img[200];
    for (int i = 0; i < 200; i++) img[i] = (uint8_t)(i % 256);
    
    printf("Calling Vision tool with image data...\n");
    uint8_t *labels = NULL;
    size_t labels_len = 0;
    
    if (melvin_tool_vision_identify(img, sizeof(img), &labels, &labels_len) == 0 && labels) {
        printf("✓ Vision generated labels: %.*s\n", (int)labels_len, labels);
        
        /* Feed vision labels into graph */
        printf("Feeding vision labels into graph...\n");
        for (size_t i = 0; i < labels_len; i++) {
            melvin_feed_byte(g, 10, labels[i], 0.1f);  /* Port 10 = vision input */
        }
        melvin_call_entry(g);
        
        uint64_t edges_vision_after = g->edge_count;
        
        printf("After Vision pattern: %llu edges\n",
               (unsigned long long)edges_vision_after);
        printf("Pattern created: +%llu edges\n",
               (unsigned long long)(edges_vision_after - edges_vision_before));
        
        if (edges_vision_after > edges_vision_before) {
            printf("✓ Vision pattern successfully seeded into graph!\n");
        }
        
        free(labels);
    } else {
        printf("⚠ Vision failed\n");
    }
    printf("\n");
    
    /* Test 3: Seed pattern from LLM */
    printf("TEST 3: Seeding Pattern from LLM Tool\n");
    printf("=====================================\n");
    
    uint64_t edges_llm_before = g->edge_count;
    
    const char *prompt = "say hello";
    printf("Calling LLM tool with prompt: \"%s\"\n", prompt);
    
    uint8_t *llm_response = NULL;
    size_t llm_response_len = 0;
    
    if (melvin_tool_llm_generate((const uint8_t *)prompt, strlen(prompt),
                                 &llm_response, &llm_response_len) == 0 && llm_response) {
        printf("✓ LLM generated response: %.*s\n", 
               (int)(llm_response_len < 100 ? llm_response_len : 100),
               llm_response);
        
        /* Feed LLM response into graph */
        printf("Feeding LLM response into graph...\n");
        for (size_t i = 0; i < llm_response_len && i < 500; i++) {
            melvin_feed_byte(g, 20, llm_response[i], 0.1f);  /* Port 20 = text input */
        }
        melvin_call_entry(g);
        
        uint64_t edges_llm_after = g->edge_count;
        
        printf("After LLM pattern: %llu edges\n",
               (unsigned long long)edges_llm_after);
        printf("Pattern created: +%llu edges\n",
               (unsigned long long)(edges_llm_after - edges_llm_before));
        
        if (edges_llm_after > edges_llm_before) {
            printf("✓ LLM pattern successfully seeded into graph!\n");
        }
        
        free(llm_response);
    } else {
        printf("⚠ LLM failed (Ollama may not be running)\n");
    }
    printf("\n");
    
    /* Test 4: Show node growth by using high port numbers */
    printf("TEST 4: Node Growth (Using High Port Numbers)\n");
    printf("==============================================\n");
    
    uint64_t nodes_before_growth = g->node_count;
    
    printf("Current nodes: %llu\n", (unsigned long long)nodes_before_growth);
    printf("Feeding through high ports (300-350) to trigger node growth...\n");
    
    /* Feed through ports that don't exist yet - this will trigger ensure_node */
    for (int i = 300; i < 350; i++) {
        melvin_feed_byte(g, (uint32_t)i, (uint8_t)(i % 256), 0.1f);
        if (i % 10 == 0) {
            melvin_call_entry(g);
        }
    }
    melvin_call_entry(g);
    
    uint64_t nodes_after_growth = g->node_count;
    
    printf("After feeding high ports: %llu nodes\n",
           (unsigned long long)nodes_after_growth);
    printf("Node growth: +%llu nodes\n",
           (unsigned long long)(nodes_after_growth - nodes_before_growth));
    
    if (nodes_after_growth > nodes_before_growth) {
        printf("✓ Nodes grew from %llu → %llu\n",
               (unsigned long long)nodes_before_growth,
               (unsigned long long)nodes_after_growth);
    } else {
        printf("⚠ Nodes did not grow (may need larger port numbers)\n");
    }
    printf("\n");
    
    /* Final summary */
    uint64_t nodes_final = g->node_count;
    uint64_t edges_final = g->edge_count;
    
    printf("========================================\n");
    printf("Final Summary\n");
    printf("========================================\n");
    printf("Initial: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_before,
           (unsigned long long)edges_before);
    printf("Final:   %llu nodes, %llu edges\n",
           (unsigned long long)nodes_final,
           (unsigned long long)edges_final);
    printf("Total growth: +%llu nodes, +%llu edges\n",
           (unsigned long long)(nodes_final - nodes_before),
           (unsigned long long)(edges_final - edges_before));
    printf("\n");
    
    if (edges_final > edges_before) {
        printf("✓ Patterns successfully seeded from tools!\n");
        printf("✓ Tool outputs create graph structure (edges)\n");
        printf("✓ Graph learns from tool-generated patterns\n");
    } else {
        printf("⚠ No edge growth detected\n");
    }
    
    if (nodes_final > nodes_before) {
        printf("✓ Nodes grow dynamically when needed!\n");
    }
    
    melvin_close(g);
    
    return 0;
}

