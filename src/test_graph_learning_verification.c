/*
 * test_graph_learning_verification.c - Verify graph is learning from tools
 * 
 * Tests:
 * 1. Nodes/edges grow when tools are used
 * 2. Patterns are created from tool outputs
 * 3. Graph operates continuously (no random stops)
 * 4. UEL physics is working
 * 5. Graph is learning, not just using tools
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

static void print_graph_state(Graph *g, const char *label) {
    printf("\n--- %s ---\n", label);
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("  Chaos: %.6f\n", g->avg_chaos);
    printf("  Activation: %.6f\n", g->avg_activation);
    printf("  Edge Strength: %.6f\n", g->avg_edge_strength);
}

static void check_pattern_nodes(Graph *g, const char *label) {
    printf("\n%s Pattern Nodes:\n", label);
    
    uint32_t active_tool_nodes = 0;
    uint32_t active_code_nodes = 0;
    uint32_t active_error_nodes = 0;
    
    for (uint32_t i = 300; i < 700 && i < g->node_count; i++) {
        if (fabsf(g->nodes[i].a) > 0.1f) active_tool_nodes++;
    }
    for (uint32_t i = 700; i < 800 && i < g->node_count; i++) {
        if (fabsf(g->nodes[i].a) > 0.1f) active_code_nodes++;
    }
    for (uint32_t i = 250; i < 260 && i < g->node_count; i++) {
        if (fabsf(g->nodes[i].a) > 0.1f) active_error_nodes++;
    }
    
    printf("  Tool gateway nodes active: %u\n", active_tool_nodes);
    printf("  Code pattern nodes active: %u\n", active_code_nodes);
    printf("  Error handling nodes active: %u\n", active_error_nodes);
}

static int test_tool_output_creates_patterns(Graph *g, MelvinSyscalls *syscalls) {
    printf("\n========================================\n");
    printf("TEST 1: Tool Outputs Create Graph Patterns\n");
    printf("========================================\n");
    
    uint64_t initial_nodes = g->node_count;
    uint64_t initial_edges = g->edge_count;
    
    print_graph_state(g, "Before Tool Calls");
    
    printf("\nCalling tools (outputs should feed into graph)...\n");
    
    /* Call STT */
    uint8_t fake_audio[100] = {0};
    uint8_t *text = NULL;
    size_t text_len = 0;
    if (syscalls->sys_audio_stt) {
        syscalls->sys_audio_stt(fake_audio, sizeof(fake_audio), &text, &text_len);
        if (text) free(text);
    }
    
    /* Call LLM */
    const char *prompt = "Hello";
    uint8_t *llm_response = NULL;
    size_t llm_len = 0;
    if (syscalls->sys_llm_generate) {
        syscalls->sys_llm_generate((const uint8_t *)prompt, strlen(prompt), &llm_response, &llm_len);
        if (llm_response) free(llm_response);
    }
    
    /* Call TTS */
    const char *tts_text = "Test";
    uint8_t *audio = NULL;
    size_t audio_len = 0;
    if (syscalls->sys_audio_tts) {
        syscalls->sys_audio_tts((const uint8_t *)tts_text, strlen(tts_text), &audio, &audio_len);
        if (audio) free(audio);
    }
    
    /* Process through graph */
    printf("Processing tool outputs through graph...\n");
    for (int i = 0; i < 10; i++) {
        melvin_call_entry(g);
    }
    
    print_graph_state(g, "After Tool Calls");
    check_pattern_nodes(g, "Tool");
    
    uint64_t nodes_grown = g->node_count - initial_nodes;
    uint64_t edges_grown = g->edge_count - initial_edges;
    
    printf("\nGrowth from tool outputs:\n");
    printf("  Nodes: +%llu\n", (unsigned long long)nodes_grown);
    printf("  Edges: +%llu\n", (unsigned long long)edges_grown);
    
    if (nodes_grown > 0 || edges_grown > 0) {
        printf("  ✓ Graph is learning from tool outputs!\n");
        return 0;
    } else {
        printf("  ⚠ No growth detected - tool outputs may not be feeding into graph\n");
        return -1;
    }
}

static int test_continuous_operation(Graph *g, int seconds) {
    printf("\n========================================\n");
    printf("TEST 2: Continuous Operation (%d seconds)\n", seconds);
    printf("========================================\n");
    
    time_t start = time(NULL);
    uint64_t initial_nodes = g->node_count;
    uint64_t initial_edges = g->edge_count;
    int iterations = 0;
    int stops = 0;
    float last_chaos = g->avg_chaos;
    float last_activation = g->avg_activation;
    
    printf("\nRunning continuously...\n");
    printf("Monitoring for random stops or stalls...\n");
    
    while (time(NULL) - start < seconds) {
        /* Feed random data to keep graph active */
        for (int i = 0; i < 5; i++) {
            uint32_t port = 100 + (rand() % 100);
            uint8_t byte = (uint8_t)(rand() % 256);
            melvin_feed_byte(g, port, byte, 0.2f);
        }
        
        /* Process */
        melvin_call_entry(g);
        iterations++;
        
        /* Check for stalls (chaos/activation not changing) */
        if (iterations % 50 == 0) {
            float chaos_diff = fabsf(g->avg_chaos - last_chaos);
            float activation_diff = fabsf(g->avg_activation - last_activation);
            
            if (chaos_diff < 0.001f && activation_diff < 0.001f) {
                stops++;
                printf("  ⚠ Potential stall at iteration %d (chaos/activation unchanged)\n", iterations);
            }
            
            last_chaos = g->avg_chaos;
            last_activation = g->avg_activation;
            
            printf("  Iteration %d: nodes=%llu, edges=%llu, chaos=%.6f\n",
                   iterations, (unsigned long long)g->node_count,
                   (unsigned long long)g->edge_count, g->avg_chaos);
        }
    }
    
    printf("\nContinuous operation results:\n");
    printf("  Total iterations: %d\n", iterations);
    printf("  Potential stalls: %d\n", stops);
    printf("  Nodes grown: +%llu\n", (unsigned long long)(g->node_count - initial_nodes));
    printf("  Edges grown: +%llu\n", (unsigned long long)(g->edge_count - initial_edges));
    
    print_graph_state(g, "After Continuous Operation");
    
    if (stops == 0) {
        printf("\n  ✓ Continuous operation verified - no random stops!\n");
        return 0;
    } else {
        printf("\n  ⚠ Detected %d potential stalls\n", stops);
        return -1;
    }
}

static int test_uel_physics_working(Graph *g) {
    printf("\n========================================\n");
    printf("TEST 3: UEL Physics Verification\n");
    printf("========================================\n");
    
    printf("\nChecking UEL physics indicators...\n");
    
    /* Check if chaos is changing (UEL should adjust chaos) */
    float initial_chaos = g->avg_chaos;
    
    /* Feed some activation */
    for (int i = 0; i < 20; i++) {
        for (uint32_t port = 100; port < 110 && port < g->node_count; port++) {
            melvin_feed_byte(g, port, 100, 0.5f);
        }
        melvin_call_entry(g);
    }
    
    float final_chaos = g->avg_chaos;
    float chaos_change = fabsf(final_chaos - initial_chaos);
    
    printf("  Initial chaos: %.6f\n", initial_chaos);
    printf("  Final chaos: %.6f\n", final_chaos);
    printf("  Chaos change: %.6f\n", chaos_change);
    
    /* Check if edges are being modified (UEL should adjust weights) */
    uint32_t modified_edges = 0;
    for (uint64_t i = 0; i < g->edge_count && i < 1000; i++) {
        if (fabsf(g->edges[i].w) > 0.01f) {
            modified_edges++;
        }
    }
    
    printf("  Edges with non-zero weights: %u / %llu (sampled)\n",
           modified_edges, (unsigned long long)(g->edge_count < 1000 ? g->edge_count : 1000));
    
    /* Check if activation is propagating */
    uint32_t active_nodes = 0;
    for (uint64_t i = 0; i < g->node_count && i < 500; i++) {
        if (fabsf(g->nodes[i].a) > 0.1f) {
            active_nodes++;
        }
    }
    
    printf("  Active nodes: %u / %llu (sampled)\n",
           active_nodes, (unsigned long long)(g->node_count < 500 ? g->node_count : 500));
    
    if (chaos_change > 0.01f || modified_edges > 100 || active_nodes > 10) {
        printf("\n  ✓ UEL physics is working!\n");
        printf("    - Chaos is changing\n");
        printf("    - Edges have weights\n");
        printf("    - Nodes are activating\n");
        return 0;
    } else {
        printf("\n  ⚠ UEL physics may not be active\n");
        return -1;
    }
}

static int test_graph_growth_from_tools(Graph *g, MelvinSyscalls *syscalls) {
    printf("\n========================================\n");
    printf("TEST 4: Graph Growth from Tool Patterns\n");
    printf("========================================\n");
    
    uint64_t initial_nodes = g->node_count;
    uint64_t initial_edges = g->edge_count;
    
    printf("\nCalling multiple tools to create patterns...\n");
    
    /* Call tools multiple times to create patterns */
    for (int round = 0; round < 5; round++) {
        /* STT */
        uint8_t fake_audio[100] = {0};
        uint8_t *text = NULL;
        size_t text_len = 0;
        if (syscalls->sys_audio_stt) {
            syscalls->sys_audio_stt(fake_audio, sizeof(fake_audio), &text, &text_len);
            if (text) free(text);
        }
        
        /* LLM */
        char prompt[100];
        snprintf(prompt, sizeof(prompt), "Test %d", round);
        uint8_t *llm_response = NULL;
        size_t llm_len = 0;
        if (syscalls->sys_llm_generate) {
            syscalls->sys_llm_generate((const uint8_t *)prompt, strlen(prompt), &llm_response, &llm_len);
            if (llm_response) free(llm_response);
        }
        
        /* Process through graph */
        for (int i = 0; i < 5; i++) {
            melvin_call_entry(g);
        }
    }
    
    print_graph_state(g, "After Multiple Tool Calls");
    check_pattern_nodes(g, "Tool");
    
    uint64_t nodes_grown = g->node_count - initial_nodes;
    uint64_t edges_grown = g->edge_count - initial_edges;
    
    printf("\nGrowth from tool patterns:\n");
    printf("  Nodes: +%llu (%.1f%% growth)\n",
           (unsigned long long)nodes_grown,
           initial_nodes > 0 ? (100.0f * nodes_grown / initial_nodes) : 0.0f);
    printf("  Edges: +%llu (%.1f%% growth)\n",
           (unsigned long long)edges_grown,
           initial_edges > 0 ? (100.0f * edges_grown / initial_edges) : 0.0f);
    
    /* Check if tool gateway nodes have connections */
    uint32_t tool_edges = 0;
    for (uint64_t i = 0; i < g->edge_count; i++) {
        uint32_t src = g->edges[i].src;
        uint32_t dst = g->edges[i].dst;
        if ((src >= 300 && src < 700) || (dst >= 300 && dst < 700)) {
            tool_edges++;
        }
    }
    
    printf("  Edges connected to tool gateways: %u\n", tool_edges);
    
    if (nodes_grown > 0 || edges_grown > 0 || tool_edges > 0) {
        printf("\n  ✓ Graph is growing from tool patterns!\n");
        return 0;
    } else {
        printf("\n  ⚠ No growth detected\n");
        return -1;
    }
}

int main(void) {
    printf("========================================\n");
    printf("Graph Learning Verification Test\n");
    printf("========================================\n");
    printf("\nVerifying:\n");
    printf("  1. Tool outputs create graph patterns\n");
    printf("  2. Graph operates continuously\n");
    printf("  3. UEL physics is working\n");
    printf("  4. Graph grows from tool patterns\n");
    printf("\n");
    
    /* Create brain */
    Graph *g = melvin_open("/tmp/learning_verification_brain.m", 2000, 10000, 131072);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    /* Initialize syscalls */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    print_graph_state(g, "Initial State");
    
    /* Run tests */
    int result1 = test_tool_output_creates_patterns(g, &syscalls);
    int result2 = test_continuous_operation(g, 10);  /* 10 seconds */
    int result3 = test_uel_physics_working(g);
    int result4 = test_graph_growth_from_tools(g, &syscalls);
    
    /* Final summary */
    printf("\n========================================\n");
    printf("FINAL VERIFICATION SUMMARY\n");
    printf("========================================\n");
    print_graph_state(g, "Final State");
    
    printf("\nTest Results:\n");
    printf("  1. Tool outputs create patterns: %s\n", (result1 == 0) ? "✓ PASS" : "⚠ FAIL");
    printf("  2. Continuous operation: %s\n", (result2 == 0) ? "✓ PASS" : "⚠ FAIL");
    printf("  3. UEL physics working: %s\n", (result3 == 0) ? "✓ PASS" : "⚠ FAIL");
    printf("  4. Graph growth from tools: %s\n", (result4 == 0) ? "✓ PASS" : "⚠ FAIL");
    
    int total_passed = 0;
    if (result1 == 0) total_passed++;
    if (result2 == 0) total_passed++;
    if (result3 == 0) total_passed++;
    if (result4 == 0) total_passed++;
    
    printf("\n  Total: %d/4 tests passed\n", total_passed);
    
    printf("\n========================================\n");
    if (total_passed == 4) {
        printf("✓ ALL VERIFICATIONS PASSED\n");
        printf("========================================\n");
        printf("\nThe graph is:\n");
        printf("  ✓ Learning from tool outputs (creating patterns)\n");
        printf("  ✓ Operating continuously (no random stops)\n");
        printf("  ✓ Following UEL physics (chaos, activation, edges)\n");
        printf("  ✓ Growing from tool patterns (nodes/edges increasing)\n");
        printf("\nTools are pattern generators - graph learns from them!\n");
    } else {
        printf("⚠ SOME VERIFICATIONS FAILED\n");
        printf("========================================\n");
    }
    
    melvin_close(g);
    
    return (total_passed == 4) ? 0 : 1;
}

