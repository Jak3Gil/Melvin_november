/*
 * test_emergent_intelligence.c - Test for emergent intelligence
 * 
 * This test measures if the system is building understanding:
 * - Starts with random/noisy patterns
 * - Feeds a corpus over many episodes
 * - Measures if patterns strengthen (chaos reduces)
 * - Tests if complex patterns form (layers emerge)
 * - Shows gradual improvement (intelligence emerging)
 * 
 * Key metric: Does the graph form layers of patterns that reduce chaos?
 * If prediction-like behavior emerges, it's organic (not coded).
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Test corpus - simple sentences that share patterns */
static const char *corpus[] = {
    "the cat sat",
    "the dog ran",
    "the cat ran",
    "the dog sat",
    "a cat sat",
    "a dog ran",
    "the cat",
    "the dog",
    "a cat",
    "a dog",
    NULL
};

/* Measure coherence: given context, does energy flow coherently? */
/* UEL doesn't "predict" - it minimizes chaos. We measure if chaos is low. */
static float test_coherence(Graph *g, const char *context, char next_byte) {
    if (!g || !context || strlen(context) == 0) return 0.0f;
    
    /* Feed context bytes */
    uint32_t in_port = 256;
    for (size_t i = 0; i < strlen(context); i++) {
        melvin_feed_byte(g, in_port, (uint8_t)context[i], 0.5f);
    }
    
    /* Let brain process (UEL minimizes chaos) */
    melvin_call_entry(g);
    
    /* Measure local incoherence (chaos) at next_byte node */
    /* UEL: chaos_i = how incoherent node i is with neighbors */
    uint32_t next_node = (uint32_t)next_byte;
    if (next_node >= g->hdr->node_count) return 1.0f; /* High chaos = bad */
    
    /* Compute local message (what neighbors say) */
    float msg = 0.0f;
    uint32_t eid = g->nodes[next_node].first_in;
    uint32_t iterations = 0;
    while (eid != UINT32_MAX && eid < g->hdr->edge_count && iterations < 1000) {
        msg += g->edges[eid].w * g->nodes[g->edges[eid].src].a;
        eid = g->edges[eid].next_in;
        iterations++;
    }
    
    /* Chaos = |a_i - msg_i| (incoherence) */
    float a_i = g->nodes[next_node].a;
    float chaos = fabsf(a_i - msg);
    
    /* Low chaos = high coherence = good (UEL is working) */
    /* Return coherence score (inverse of chaos) */
    return 1.0f / (1.0f + chaos);
}

/* Test if system reduced chaos (UEL working) */
static void test_chaos_reduction(Graph *g, int episode) {
    printf("\n--- Episode %d: Chaos Reduction Test (UEL) ---\n", episode);
    
    struct {
        const char *context;
        char next;
        const char *description;
    } tests[] = {
        {"the c", 'a', "Context 'the c' → 'a' (chaos should be low)"},
        {"the d", 'o', "Context 'the d' → 'o' (chaos should be low)"},
        {"a c", 'a', "Context 'a c' → 'a' (chaos should be low)"},
        {"a d", 'o', "Context 'a d' → 'o' (chaos should be low)"},
        {"the cat ", 's', "Context 'the cat ' → 's' (chaos should be low)"},
        {"the dog ", 'r', "Context 'the dog ' → 'r' (chaos should be low)"},
    };
    
    float total_coherence = 0.0f;
    int low_chaos = 0;
    
    for (size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) {
        float coherence = test_coherence(g, tests[i].context, tests[i].next);
        total_coherence += coherence;
        
        /* High coherence = low chaos = UEL working */
        printf("  %s: coherence=%.4f", tests[i].description, coherence);
        if (coherence > 0.5f) {  /* Low chaos */
            printf(" ✓ (low chaos)");
            low_chaos++;
        }
        printf("\n");
    }
    
    float avg_coherence = total_coherence / (sizeof(tests)/sizeof(tests[0]));
    printf("  Average coherence: %.4f (%d/%zu low-chaos)\n", 
           avg_coherence, low_chaos, sizeof(tests)/sizeof(tests[0]));
    
    if (avg_coherence > 0.7f) {
        printf("  [EMERGING] Low chaos - UEL is creating coherent flows!\n");
    } else if (avg_coherence > 0.3f) {
        printf("  [LEARNING] Chaos reducing - patterns forming\n");
    } else {
        printf("  [RANDOM] High chaos - no coherent patterns yet\n");
    }
}

/* Measure total chaos in graph (UEL minimizes this) */
static float measure_total_chaos(Graph *g) {
    if (!g || g->hdr->node_count == 0) return 0.0f;
    
    float total_chaos = 0.0f;
    
    /* Compute local messages */
    float *msg = calloc(g->hdr->node_count, sizeof(float));
    if (!msg) return 0.0f;
    
    for (size_t i = 0; i < g->hdr->node_count; i++) {
        uint32_t eid = g->nodes[i].first_in;
        while (eid != UINT32_MAX && eid < g->hdr->edge_count) {
            msg[i] += g->edges[eid].w * g->nodes[g->edges[eid].src].a;
            eid = g->edges[eid].next_in;
        }
    }
    
    /* Chaos = Σ_i |a_i - msg_i|² (incoherence) */
    for (size_t i = 0; i < g->hdr->node_count; i++) {
        float chaos_i = (g->nodes[i].a - msg[i]) * (g->nodes[i].a - msg[i]);
        total_chaos += chaos_i;
    }
    
    free(msg);
    return total_chaos;
}

/* Measure pattern layers: how many multi-hop paths exist? */
static int measure_pattern_layers(Graph *g) {
    if (!g || g->hdr->edge_count == 0) return 0;
    
    /* Count nodes that are part of multi-hop paths */
    /* A node is in a layer if it has both incoming and outgoing strong edges */
    int layered_nodes = 0;
    
    for (size_t i = 0; i < g->hdr->node_count; i++) {
        int has_strong_in = 0;
        int has_strong_out = 0;
        
        /* Check incoming edges */
        uint32_t eid = g->nodes[i].first_in;
        while (eid != UINT32_MAX && eid < g->hdr->edge_count) {
            if (fabsf(g->edges[eid].w) > 0.2f) {
                has_strong_in = 1;
                break;
            }
            eid = g->edges[eid].next_in;
        }
        
        /* Check outgoing edges */
        eid = g->nodes[i].first_out;
        while (eid != UINT32_MAX && eid < g->hdr->edge_count) {
            if (fabsf(g->edges[eid].w) > 0.2f) {
                has_strong_out = 1;
                break;
            }
            eid = g->edges[eid].next_out;
        }
        
        if (has_strong_in && has_strong_out) {
            layered_nodes++;  /* Part of a multi-hop pattern */
        }
    }
    
    return layered_nodes;
}

int main(void) {
    printf("=== Emergent Intelligence Test ===\n\n");
    printf("This test measures if intelligence is emerging:\n");
    printf("  - Starts with high chaos (random patterns)\n");
    printf("  - Feeds corpus over many episodes\n");
    printf("  - UEL minimizes chaos (free energy)\n");
    printf("  - Measures if chaos reduces (coherent patterns form)\n\n");
    
    const char *brain_path = "intelligence_test.m";
    Graph *g = melvin_open(brain_path, 2000, 50000, 131072);
    if (!g) {
        printf("FAIL: Could not open %s\n", brain_path);
        return 1;
    }
    
    printf("[OK] Opened %s\n", brain_path);
    
    /* Set up syscalls */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    uint32_t in_port = 256;
    
    /* Initial state */
    printf("\n--- Initial State ---\n");
    float initial_chaos = measure_total_chaos(g);
    int initial_layers = measure_pattern_layers(g);
    printf("Initial total chaos: %.4f\n", initial_chaos);
    printf("Pattern layers (multi-hop nodes): %d\n", initial_layers);
    printf("Edges: %llu\n", (unsigned long long)g->hdr->edge_count);
    
    /* Training phases */
    const int phases = 5;
    const int episodes_per_phase = 20;
    
    printf("\n--- Training: %d phases × %d episodes ---\n", phases, episodes_per_phase);
    printf("Feeding corpus: 'the cat sat', 'the dog ran', etc.\n\n");
    
    for (int phase = 0; phase < phases; phase++) {
        printf("Phase %d/%d:\n", phase + 1, phases);
        
        /* Feed corpus multiple times */
        for (int ep = 0; ep < episodes_per_phase; ep++) {
            /* Pick random sentence from corpus */
            int corpus_idx = 0;
            while (corpus[corpus_idx] != NULL) corpus_idx++;
            int rand_idx = rand() % corpus_idx;
            const char *sentence = corpus[rand_idx];
            
            /* Feed sentence */
            for (size_t i = 0; sentence[i]; i++) {
                melvin_feed_byte(g, in_port, (uint8_t)sentence[i], 1.0f);
            }
            
            /* Let brain process */
            melvin_call_entry(g);
        }
        
        /* Measure UEL working: chaos should decrease, patterns should form */
        float current_chaos = measure_total_chaos(g);
        int current_layers = measure_pattern_layers(g);
        printf("  Total chaos: %.4f (was %.4f, change: %.4f)\n", 
               current_chaos, initial_chaos, initial_chaos - current_chaos);
        printf("  Pattern layers: %d (was %d)\n", current_layers, initial_layers);
        printf("  Edges: %llu\n", (unsigned long long)g->hdr->edge_count);
        
        /* Test chaos reduction every phase */
        if ((phase + 1) % 2 == 0 || phase == phases - 1) {
            test_chaos_reduction(g, (phase + 1) * episodes_per_phase);
        }
        
        initial_chaos = current_chaos;
        initial_layers = current_layers;
    }
    
    /* Final assessment */
    printf("\n=== Final Assessment ===\n");
    float final_chaos = measure_total_chaos(g);
    int final_layers = measure_pattern_layers(g);
    printf("Final total chaos: %.4f\n", final_chaos);
    printf("Final pattern layers: %d\n", final_layers);
    printf("Total edges: %llu\n", (unsigned long long)g->hdr->edge_count);
    
    /* Final chaos reduction test */
    test_chaos_reduction(g, phases * episodes_per_phase);
    
    printf("\n=== Emergent Intelligence Metrics ===\n");
    
    /* Chaos reduction = UEL working */
    float chaos_reduction = initial_chaos - final_chaos;
    printf("Chaos reduced by: %.4f\n", chaos_reduction);
    if (chaos_reduction > 0.1f) {
        printf("[EMERGING] Chaos reducing - UEL is working!\n");
    }
    
    /* Pattern layers = complex patterns forming */
    int layer_growth = final_layers - initial_layers;
    printf("Pattern layers increased by: %d\n", layer_growth);
    if (layer_growth > 5) {
        printf("[EMERGING] Multi-hop patterns forming - layers emerging!\n");
    }
    
    /* Strong patterns = organic constraints */
    int strong_patterns = 0;
    for (size_t i = 0; i < g->hdr->edge_count; i++) {
        if (fabsf(g->edges[i].w) > 0.3f) {
            strong_patterns++;
        }
    }
    printf("Strong patterns (|w| > 0.3): %d\n", strong_patterns);
    
    if (strong_patterns > 10 && layer_growth > 5) {
        printf("[INTELLIGENCE] Complex patterns + layers - organic constraints emerging!\n");
        printf("  If prediction-like behavior appears, it emerged organically.\n");
        printf("  We didn't code it - UEL found it's energy-efficient.\n");
    }
    
    melvin_sync(g);
    melvin_close(g);
    
    printf("\n[OK] Brain saved to %s\n", brain_path);
    printf("\nKey insight: Each byte helps build understanding.\n");
    printf("Early episodes: random, high chaos\n");
    printf("Later episodes: patterns form, chaos reduces, layers emerge\n");
    printf("If prediction-like behavior appears, it emerged organically.\n");
    printf("We didn't code it - UEL found it's energy-efficient.\n");
    
    return 0;
}

