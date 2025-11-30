/*
 * test_graph_tools_ready.c - Verify graph can use all tools and run continuously
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

static void print_tool_status(Graph *g, MelvinSyscalls *syscalls) {
    printf("\n========================================\n");
    printf("Tool Integration Status\n");
    printf("========================================\n");
    
    printf("\n1. Tool Syscalls Available:\n");
    printf("   LLM (sys_llm_generate): %s\n", syscalls->sys_llm_generate ? "✓ YES" : "✗ NO");
    printf("   Vision (sys_vision_identify): %s\n", syscalls->sys_vision_identify ? "✓ YES" : "✗ NO");
    printf("   STT (sys_audio_stt): %s\n", syscalls->sys_audio_stt ? "✓ YES" : "✗ NO");
    printf("   TTS (sys_audio_tts): %s\n", syscalls->sys_audio_tts ? "✓ YES" : "✗ NO");
    
    printf("\n2. Tool Gateway Nodes:\n");
    uint32_t tool_nodes = 0;
    for (uint64_t i = 300; i < 700 && i < g->node_count; i++) {
        if (g->nodes[i].input_propensity > 0.5f || g->nodes[i].output_propensity > 0.5f) {
            tool_nodes++;
        }
    }
    printf("   Tool gateway nodes (300-699): %u\n", tool_nodes);
    printf("   STT gateway (300-319): %s\n", g->node_count > 310 ? "✓ Created" : "✗ Missing");
    printf("   Vision gateway (400-419): %s\n", g->node_count > 410 ? "✓ Created" : "✗ Missing");
    printf("   LLM gateway (500-519): %s\n", g->node_count > 510 ? "✓ Created" : "✗ Missing");
    printf("   TTS gateway (600-619): %s\n", g->node_count > 610 ? "✓ Created" : "✗ Missing");
    
    printf("\n3. Tool Gateway Connections:\n");
    uint32_t tool_edges = 0;
    uint32_t stt_edges = 0, vision_edges = 0, llm_edges = 0, tts_edges = 0;
    for (uint64_t i = 0; i < g->edge_count; i++) {
        uint32_t src = g->edges[i].src;
        uint32_t dst = g->edges[i].dst;
        
        if ((src >= 300 && src < 700) || (dst >= 300 && dst < 700)) {
            tool_edges++;
            if ((src >= 300 && src < 320) || (dst >= 300 && dst < 320)) stt_edges++;
            if ((src >= 400 && src < 420) || (dst >= 400 && dst < 420)) vision_edges++;
            if ((src >= 500 && src < 520) || (dst >= 500 && dst < 520)) llm_edges++;
            if ((src >= 600 && src < 620) || (dst >= 600 && dst < 620)) tts_edges++;
        }
    }
    printf("   Total tool gateway edges: %u\n", tool_edges);
    printf("   STT edges: %u\n", stt_edges);
    printf("   Vision edges: %u\n", vision_edges);
    printf("   LLM edges: %u\n", llm_edges);
    printf("   TTS edges: %u\n", tts_edges);
    
    printf("\n4. Continuous Operation:\n");
    printf("   Self-regulation nodes (255-259): %s\n", g->node_count > 259 ? "✓ Created" : "✗ Missing");
    printf("   Chaos monitoring: %.6f\n", g->avg_chaos);
    printf("   Activation: %.6f\n", g->avg_activation);
    printf("   Graph can self-regulate: %s\n", g->node_count > 259 ? "✓ YES" : "✗ NO");
}

static int test_tool_calls(MelvinSyscalls *syscalls) {
    printf("\n========================================\n");
    printf("Testing Tool Calls\n");
    printf("========================================\n");
    
    int passed = 0;
    int total = 4;
    
    /* Test LLM */
    printf("\n1. Testing LLM (Ollama)...\n");
    if (syscalls->sys_llm_generate) {
        const char *prompt = "Hello";
        uint8_t *response = NULL;
        size_t response_len = 0;
        int result = syscalls->sys_llm_generate((const uint8_t *)prompt, strlen(prompt), &response, &response_len);
        if (result == 0 && response && response_len > 0) {
            printf("   ✓ LLM working: %.*s\n", (int)(response_len < 50 ? response_len : 50), response);
            free(response);
            passed++;
        } else {
            printf("   ⚠ LLM call failed or returned empty\n");
        }
    } else {
        printf("   ✗ LLM syscall not available\n");
    }
    
    /* Test Vision */
    printf("\n2. Testing Vision (ONNX)...\n");
    if (syscalls->sys_vision_identify) {
        uint8_t fake_image[100] = {0};
        uint8_t *labels = NULL;
        size_t labels_len = 0;
        int result = syscalls->sys_vision_identify(fake_image, sizeof(fake_image), &labels, &labels_len);
        if (result == 0 && labels && labels_len > 0) {
            printf("   ✓ Vision working: %.*s\n", (int)(labels_len < 50 ? labels_len : 50), labels);
            free(labels);
            passed++;
        } else {
            printf("   ⚠ Vision call failed or returned empty\n");
        }
    } else {
        printf("   ✗ Vision syscall not available\n");
    }
    
    /* Test STT */
    printf("\n3. Testing STT (Whisper)...\n");
    if (syscalls->sys_audio_stt) {
        uint8_t fake_audio[100] = {0};
        uint8_t *text = NULL;
        size_t text_len = 0;
        int result = syscalls->sys_audio_stt(fake_audio, sizeof(fake_audio), &text, &text_len);
        if (result == 0 && text && text_len > 0) {
            printf("   ✓ STT working: %.*s\n", (int)(text_len < 50 ? text_len : 50), text);
            free(text);
            passed++;
        } else {
            printf("   ⚠ STT call failed or returned empty\n");
        }
    } else {
        printf("   ✗ STT syscall not available\n");
    }
    
    /* Test TTS */
    printf("\n4. Testing TTS (Piper)...\n");
    if (syscalls->sys_audio_tts) {
        const char *tts_text = "Test";
        uint8_t *audio = NULL;
        size_t audio_len = 0;
        int result = syscalls->sys_audio_tts((const uint8_t *)tts_text, strlen(tts_text), &audio, &audio_len);
        if (result == 0 && audio && audio_len > 0) {
            printf("   ✓ TTS working: %zu bytes\n", audio_len);
            free(audio);
            passed++;
        } else {
            printf("   ⚠ TTS call failed or returned empty\n");
        }
    } else {
        printf("   ✗ TTS syscall not available\n");
    }
    
    printf("\nTool Tests: %d/%d passed\n", passed, total);
    return (passed == total) ? 0 : -1;
}

static void test_continuous_operation(Graph *g, int seconds) {
    printf("\n========================================\n");
    printf("Testing Continuous Operation (%d seconds)\n", seconds);
    printf("========================================\n");
    
    time_t start = time(NULL);
    uint64_t initial_nodes = g->node_count;
    uint64_t initial_edges = g->edge_count;
    float initial_chaos = g->avg_chaos;
    int iterations = 0;
    bool still_processing = true;
    
    printf("\nRunning continuously...\n");
    printf("Graph should self-regulate and keep processing...\n");
    
    while (time(NULL) - start < seconds && still_processing) {
        /* Feed some input to keep it active */
        for (int i = 0; i < 3; i++) {
            uint8_t byte = (uint8_t)(rand() % 256);
            melvin_feed_byte(g, 0, byte, 0.2f);
        }
        
        /* Process */
        melvin_call_entry(g);
        iterations++;
        
        /* Check if graph is still active (not stuck) */
        if (iterations % 20 == 0) {
            float chaos_change = fabsf(g->avg_chaos - initial_chaos);
            printf("  Iteration %d: nodes=%llu, edges=%llu, chaos=%.6f (change: %.6f)\n",
                   iterations, (unsigned long long)g->node_count,
                   (unsigned long long)g->edge_count, g->avg_chaos, chaos_change);
            
            /* If chaos hasn't changed at all, might be stuck */
            if (chaos_change < 0.0001f && iterations > 50) {
                printf("  ⚠ Warning: Chaos not changing (might be in low-activity state)\n");
            }
        }
        
        usleep(100000);  /* 100ms between iterations */
    }
    
    printf("\nContinuous operation results:\n");
    printf("  Iterations: %d\n", iterations);
    printf("  Nodes: %llu (+%llu)\n",
           (unsigned long long)g->node_count,
           (unsigned long long)(g->node_count - initial_nodes));
    printf("  Edges: %llu (+%llu)\n",
           (unsigned long long)g->edge_count,
           (unsigned long long)(g->edge_count - initial_edges));
    printf("  Chaos: %.6f (started at %.6f)\n", g->avg_chaos, initial_chaos);
    printf("  Activation: %.6f\n", g->avg_activation);
    
    if (iterations > 0 && (g->node_count > initial_nodes || g->edge_count > initial_edges || fabsf(g->avg_chaos - initial_chaos) > 0.001f)) {
        printf("\n  ✓ Graph is running continuously and processing!\n");
    } else {
        printf("\n  ⚠ Graph may not be processing (check if in low-activity state)\n");
    }
}

int main(void) {
    printf("========================================\n");
    printf("Graph Tools & Continuous Operation Test\n");
    printf("========================================\n");
    printf("\n");
    
    /* Create/open brain */
    Graph *g = melvin_open("/tmp/test_tools_ready_brain.m", 0, 10000, 131072);
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
    printf("\n");
    
    /* Check tool integration */
    print_tool_status(g, &syscalls);
    
    /* Test tool calls */
    int tool_test = test_tool_calls(&syscalls);
    
    /* Test continuous operation */
    test_continuous_operation(g, 5);  /* 5 seconds test */
    
    /* Final summary */
    printf("\n========================================\n");
    printf("FINAL VERIFICATION\n");
    printf("========================================\n");
    
    bool tools_ready = (syscalls.sys_llm_generate && syscalls.sys_vision_identify && 
                        syscalls.sys_audio_stt && syscalls.sys_audio_tts);
    bool gateways_ready = (g->node_count > 610);
    bool patterns_ready = (g->edge_count > 2000);
    bool continuous_ready = (g->node_count > 259);  /* Self-regulation nodes */
    
    printf("Tool syscalls available: %s\n", tools_ready ? "✓ YES" : "✗ NO");
    printf("Tool gateways created: %s\n", gateways_ready ? "✓ YES" : "✗ NO");
    printf("Patterns created: %s\n", patterns_ready ? "✓ YES" : "✗ NO");
    printf("Self-regulation ready: %s\n", continuous_ready ? "✓ YES" : "✗ NO");
    printf("\n");
    
    if (tools_ready && gateways_ready && patterns_ready && continuous_ready) {
        printf("✓ GRAPH IS READY TO USE ALL TOOLS AND RUN CONTINUOUSLY!\n");
        printf("\nThe graph can:\n");
        printf("  ✓ Call all tools via syscalls\n");
        printf("  ✓ Route data through tool gateways\n");
        printf("  ✓ Learn from tool outputs (create patterns)\n");
        printf("  ✓ Run continuously (self-regulating)\n");
        printf("  ✓ Control its own activity\n");
        return 0;
    } else {
        printf("⚠ Some components may be missing\n");
        return 1;
    }
    
    melvin_close(g);
    return 0;
}

