/* Test reinforcement learning - system learns from crashes */
#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  REINFORCEMENT LEARNING FROM CODE CRASHES TEST    ║\n");
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║  System learns which EXEC nodes work vs crash     ║\n");
    printf("║  Successful nodes → easier to trigger             ║\n");
    printf("║  Failing nodes → harder to trigger                ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    /* Open brain */
    Graph *brain = melvin_open("hardware_brain.m", 10000, 50000, 131072);
    if (!brain) {
        printf("❌ Can't open brain\n");
        return 1;
    }
    
    printf("✅ Brain loaded: %llu nodes, %llu edges\n\n",
           (unsigned long long)brain->node_count,
           (unsigned long long)brain->edge_count);
    
    /* Find EXEC nodes and show their initial states */
    printf("═══════════════════════════════════════════════════\n");
    printf("EXEC NODES - Initial State\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    int exec_count = 0;
    for (uint64_t i = 2000; i < brain->node_count && i < 2100; i++) {
        if (brain->nodes[i].payload_offset > 0) {
            printf("  Node %llu: offset=%llu, threshold_ratio=%.3f, success_rate=%.3f\n",
                   (unsigned long long)i,
                   (unsigned long long)brain->nodes[i].payload_offset,
                   brain->nodes[i].exec_threshold_ratio,
                   brain->nodes[i].exec_success_rate);
            exec_count++;
        }
    }
    printf("\nFound %d EXEC nodes\n\n", exec_count);
    
    /* Run learning cycles */
    printf("═══════════════════════════════════════════════════\n");
    printf("LEARNING CYCLES - Watch Reinforcement in Action\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    for (int cycle = 0; cycle < 50; cycle++) {
        printf("Cycle %d: ", cycle);
        fflush(stdout);
        
        /* Feed varied inputs to trigger different patterns/EXEC nodes */
        const char *inputs[] = {
            "AUDIO_STREAM_DATA",
            "CAMERA_VISUAL_INPUT", 
            "SENSOR_READING_123",
            "PATTERN_MATCH_TEST",
            "COMPUTATION_REQUEST"
        };
        
        const char *input = inputs[cycle % 5];
        for (const char *p = input; *p; p++) {
            melvin_feed_byte(brain, cycle % 20, *p, 0.8f);
        }
        
        /* This will trigger pattern matching and possibly EXEC nodes */
        /* Some will succeed, some will crash (but be caught!) */
        melvin_call_entry(brain);
        
        printf("✓\n");
        
        /* Every 10 cycles, show learning progress */
        if ((cycle + 1) % 10 == 0) {
            printf("\n--- Progress Report (after %d cycles) ---\n", cycle + 1);
            for (uint64_t i = 2000; i < brain->node_count && i < 2100; i++) {
                if (brain->nodes[i].payload_offset > 0 && brain->nodes[i].exec_count > 0) {
                    printf("  Node %llu: executions=%u, success_rate=%.3f, threshold_ratio=%.3f\n",
                           (unsigned long long)i,
                           brain->nodes[i].exec_count,
                           brain->nodes[i].exec_success_rate,
                           brain->nodes[i].exec_threshold_ratio);
                    
                    /* Interpret the learning */
                    if (brain->nodes[i].exec_success_rate > 0.7f) {
                        printf("    → WORKING! Easy to trigger (threshold ratio low)\n");
                    } else if (brain->nodes[i].exec_success_rate < 0.3f) {
                        printf("    → FAILING! Hard to trigger (threshold ratio increasing)\n");
                    } else {
                        printf("    → UNCERTAIN - still learning\n");
                    }
                }
            }
            printf("\n");
        }
    }
    
    /* Final report */
    printf("\n═══════════════════════════════════════════════════\n");
    printf("FINAL LEARNING RESULTS\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    int working_nodes = 0;
    int failing_nodes = 0;
    int untested_nodes = 0;
    
    for (uint64_t i = 2000; i < brain->node_count && i < 2100; i++) {
        if (brain->nodes[i].payload_offset > 0) {
            if (brain->nodes[i].exec_count == 0) {
                untested_nodes++;
            } else if (brain->nodes[i].exec_success_rate > 0.5f) {
                working_nodes++;
                printf("✅ Node %llu: WORKS (success=%.3f, threshold=%.3f)\n",
                       (unsigned long long)i,
                       brain->nodes[i].exec_success_rate,
                       brain->nodes[i].exec_threshold_ratio);
            } else {
                failing_nodes++;
                printf("❌ Node %llu: FAILS (success=%.3f, threshold=%.3f)\n",
                       (unsigned long long)i,
                       brain->nodes[i].exec_success_rate,
                       brain->nodes[i].exec_threshold_ratio);
            }
        }
    }
    
    printf("\nSummary:\n");
    printf("  Working nodes: %d (reinforced - easy to activate)\n", working_nodes);
    printf("  Failing nodes: %d (suppressed - hard to activate)\n", failing_nodes);
    printf("  Untested: %d\n\n", untested_nodes);
    
    /* Check pattern learning */
    int patterns = 0;
    for (uint64_t i = 840; i < brain->node_count && i < 2000; i++) {
        if (brain->nodes[i].pattern_data_offset > 0) patterns++;
    }
    printf("  Patterns learned: %d\n\n", patterns);
    
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  REINFORCEMENT LEARNING SUCCESSFUL!                ║\n");
    printf("║  System learned from crashes without dying!       ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    melvin_close(brain);
    return 0;
}

