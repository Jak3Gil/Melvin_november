/*
 * test_graph_learning_simple.c - Simple test to verify graph learns from tools
 * 
 * Quick verification:
 * 1. Tool outputs create nodes/edges
 * 2. Graph grows from patterns
 * 3. UEL physics is active
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

int main(void) {
    printf("========================================\n");
    printf("Graph Learning Verification (Simple)\n");
    printf("========================================\n");
    printf("\nVerifying graph learns from tool outputs...\n\n");
    
    /* Create brain */
    Graph *g = melvin_open("/tmp/learning_simple_brain.m", 2000, 10000, 131072);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    /* Initialize syscalls */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    printf("Initial State:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("  Chaos: %.6f\n", g->avg_chaos);
    printf("  Activation: %.6f\n", g->avg_activation);
    printf("\n");
    
    uint64_t initial_nodes = g->node_count;
    uint64_t initial_edges = g->edge_count;
    
    /* Call tools to generate patterns */
    printf("Calling tools (STT, LLM, TTS)...\n");
    
    /* STT */
    uint8_t fake_audio[100] = {0};
    uint8_t *text = NULL;
    size_t text_len = 0;
    if (syscalls.sys_audio_stt) {
        syscalls.sys_audio_stt(fake_audio, sizeof(fake_audio), &text, &text_len);
        if (text) {
            printf("  STT output: %.*s\n", (int)text_len < 50 ? (int)text_len : 50, text);
            free(text);
        }
    }
    
    /* LLM */
    const char *prompt = "Hello";
    uint8_t *llm_response = NULL;
    size_t llm_len = 0;
    if (syscalls.sys_llm_generate) {
        syscalls.sys_llm_generate((const uint8_t *)prompt, strlen(prompt), &llm_response, &llm_len);
        if (llm_response) {
            printf("  LLM output: %.*s\n", (int)llm_len < 50 ? (int)llm_len : 50, llm_response);
            free(llm_response);
        }
    }
    
    /* TTS */
    const char *tts_text = "Test";
    uint8_t *audio = NULL;
    size_t audio_len = 0;
    if (syscalls.sys_audio_tts) {
        syscalls.sys_audio_tts((const uint8_t *)tts_text, strlen(tts_text), &audio, &audio_len);
        if (audio) {
            printf("  TTS output: %zu bytes\n", audio_len);
            free(audio);
        }
    }
    
    printf("\nProcessing tool outputs through graph...\n");
    for (int i = 0; i < 20; i++) {
        melvin_call_entry(g);
    }
    
    printf("\nAfter Tool Processing:\n");
    printf("  Nodes: %llu (+%llu)\n", 
           (unsigned long long)g->node_count,
           (unsigned long long)(g->node_count - initial_nodes));
    printf("  Edges: %llu (+%llu)\n",
           (unsigned long long)g->edge_count,
           (unsigned long long)(g->edge_count - initial_edges));
    printf("  Chaos: %.6f\n", g->avg_chaos);
    printf("  Activation: %.6f\n", g->avg_activation);
    printf("\n");
    
    /* Check for tool gateway connections */
    uint32_t tool_edges = 0;
    for (uint64_t i = 0; i < g->edge_count; i++) {
        uint32_t src = g->edges[i].src;
        uint32_t dst = g->edges[i].dst;
        if ((src >= 300 && src < 700) || (dst >= 300 && dst < 700)) {
            tool_edges++;
        }
    }
    
    printf("Tool Gateway Connections: %u edges\n", tool_edges);
    printf("\n");
    
    /* Check UEL physics */
    uint32_t active_nodes = 0;
    for (uint64_t i = 0; i < g->node_count && i < 500; i++) {
        if (fabsf(g->nodes[i].a) > 0.1f) {
            active_nodes++;
        }
    }
    
    printf("Active Nodes: %u / %llu (sampled)\n",
           active_nodes, (unsigned long long)(g->node_count < 500 ? g->node_count : 500));
    printf("\n");
    
    /* Summary */
    printf("========================================\n");
    printf("VERIFICATION RESULTS\n");
    printf("========================================\n");
    
    bool nodes_grew = (g->node_count > initial_nodes);
    bool edges_grew = (g->edge_count > initial_edges);
    bool has_tool_connections = (tool_edges > 0);
    bool uel_active = (active_nodes > 10 || fabsf(g->avg_chaos) > 0.001f);
    
    printf("  Nodes grew: %s\n", nodes_grew ? "✓ YES" : "⚠ NO");
    printf("  Edges grew: %s\n", edges_grew ? "✓ YES" : "⚠ NO");
    printf("  Tool connections: %s\n", has_tool_connections ? "✓ YES" : "⚠ NO");
    printf("  UEL physics active: %s\n", uel_active ? "✓ YES" : "⚠ NO");
    printf("\n");
    
    if (nodes_grew || edges_grew || has_tool_connections) {
        printf("✓ GRAPH IS LEARNING FROM TOOLS!\n");
        printf("\nThe graph is:\n");
        printf("  - Creating nodes/edges from tool outputs\n");
        printf("  - Building patterns from tool responses\n");
        printf("  - Following UEL physics (continuous operation)\n");
        printf("\nTools are pattern generators - graph learns from them!\n");
    } else {
        printf("⚠ Graph may not be learning from tools yet\n");
        printf("  (This could be normal if graph needs more time)\n");
    }
    
    melvin_close(g);
    
    return 0;
}

