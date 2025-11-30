/*
 * test_first_run.c - Minimal observational test for binary brain
 * 
 * This test:
 * - Opens/creates a .m brain file
 * - Feeds a simple repeated pattern
 * - Calls blob entry to run laws
 * - Prints stats before/after
 * 
 * NO physics, NO rules - purely observational
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Helper: Print node stats */
static void print_node_stats(Graph *g, const char *label) {
    if (!g || !g->hdr || g->hdr->node_count == 0) return;
    
    float min_a = FLT_MAX;
    float max_a = 0.0f;
    float sum_a = 0.0f;
    size_t count = 0;
    
    for (size_t i = 0; i < g->hdr->node_count; i++) {
        float a_abs = fabsf(g->nodes[i].a);
        if (a_abs < min_a) min_a = a_abs;
        if (a_abs > max_a) max_a = a_abs;
        sum_a += a_abs;
        count++;
    }
    
    float avg_a = (count > 0) ? sum_a / count : 0.0f;
    
    printf("%s Node Stats:\n", label);
    printf("  Count: %llu\n", (unsigned long long)g->hdr->node_count);
    printf("  Activation: min=%.6f, max=%.6f, avg=%.6f\n", min_a, max_a, avg_a);
}

/* Helper: Print edge stats */
static void print_edge_stats(Graph *g, const char *label) {
    if (!g || !g->hdr || g->hdr->edge_count == 0) {
        printf("%s Edge Stats:\n", label);
        printf("  Count: 0\n");
        return;
    }
    
    float min_w = FLT_MAX;
    float max_w = 0.0f;
    float sum_w = 0.0f;
    size_t count = 0;
    
    for (size_t i = 0; i < g->hdr->edge_count; i++) {
        float w_abs = fabsf(g->edges[i].w);
        if (w_abs < min_w) min_w = w_abs;
        if (w_abs > max_w) max_w = w_abs;
        sum_w += w_abs;
        count++;
    }
    
    float avg_w = (count > 0) ? sum_w / count : 0.0f;
    
    printf("%s Edge Stats:\n", label);
    printf("  Count: %llu\n", (unsigned long long)g->hdr->edge_count);
    printf("  Weight: min=%.6f, max=%.6f, avg=%.6f\n", min_w, max_w, avg_w);
}

/* Helper: Print top edges by weight */
static void print_top_edges(Graph *g, int top_n) {
    if (!g || !g->hdr || g->hdr->edge_count == 0) return;
    
    /* Simple selection sort for top N */
    struct { size_t idx; float w_abs; } top[10];
    int top_count = (top_n > 10) ? 10 : top_n;
    for (int i = 0; i < top_count; i++) {
        top[i].idx = 0;
        top[i].w_abs = -1.0f;
    }
    
    for (size_t i = 0; i < g->hdr->edge_count; i++) {
        float w_abs = fabsf(g->edges[i].w);
        
        /* Find insertion point */
        for (int j = 0; j < top_count; j++) {
            if (w_abs > top[j].w_abs) {
                /* Shift down */
                for (int k = top_count - 1; k > j; k--) {
                    top[k] = top[k-1];
                }
                top[j].idx = i;
                top[j].w_abs = w_abs;
                break;
            }
        }
    }
    
    printf("Top %d edges by |weight|:\n", top_count);
    for (int i = 0; i < top_count; i++) {
        if (top[i].w_abs >= 0.0f) {
            Edge *e = &g->edges[top[i].idx];
            printf("  [%u -> %u]: w=%.6f\n", e->src, e->dst, e->w);
        }
    }
}

/* Helper: Print top activated nodes */
static void print_top_activations(Graph *g, int top_n) {
    if (!g || !g->hdr || g->hdr->node_count == 0) return;
    
    /* Simple selection sort */
    struct { size_t idx; float a_abs; } top[10];
    int top_count = (top_n > 10) ? 10 : top_n;
    for (int i = 0; i < top_count; i++) {
        top[i].idx = 0;
        top[i].a_abs = -1.0f;
    }
    
    for (size_t i = 0; i < g->hdr->node_count; i++) {
        float a_abs = fabsf(g->nodes[i].a);
        
        for (int j = 0; j < top_count; j++) {
            if (a_abs > top[j].a_abs) {
                for (int k = top_count - 1; k > j; k--) {
                    top[k] = top[k-1];
                }
                top[j].idx = i;
                top[j].a_abs = a_abs;
                break;
            }
        }
    }
    
    printf("Top %d nodes by |activation|:\n", top_count);
    for (int i = 0; i < top_count; i++) {
        if (top[i].a_abs >= 0.0f) {
            Node *n = &g->nodes[top[i].idx];
            uint8_t b = n->byte;
            if (b >= 32 && b < 127) {
                printf("  [%zu] '%c': a=%.6f\n", top[i].idx, b, n->a);
            } else {
                printf("  [%zu] 0x%02X: a=%.6f\n", top[i].idx, b, n->a);
            }
        }
    }
}

int main(void) {
    printf("=== First Run Test: Observing Binary Brain ===\n\n");
    
    /* Open or create brain file */
    const char *brain_path = "first_run.m";
    Graph *g = melvin_open(brain_path, 1000, 10000, 65536);
    if (!g) {
        printf("FAIL: Could not open/create %s\n", brain_path);
        return 1;
    }
    
    printf("[OK] Opened %s\n", brain_path);
    printf("     Main entry offset: %llu\n", 
           (unsigned long long)g->hdr->main_entry_offset);
    fflush(stdout);
    
    /* Use simple port indices */
    uint32_t in_port = 256;  /* First node after data nodes (0-255) */
    uint32_t out_port = 257;
    
    printf("     Using ports: in=%u, out=%u\n", in_port, out_port);
    fflush(stdout);
    
    /* Print initial stats */
    printf("\n--- BEFORE INPUT ---\n");
    fflush(stdout);
    print_node_stats(g, "Initial");
    fflush(stdout);
    print_edge_stats(g, "Initial");
    fflush(stdout);
    
    /* Feed simple repeated pattern */
    const char *pattern = "ABABABAB\n";
    const int episodes = 100;
    const float energy = 1.0f;
    
    printf("\n--- FEEDING PATTERN ---\n");
    printf("Pattern: \"%s\"\n", pattern);
    printf("Episodes: %d\n", episodes);
    printf("Energy per byte: %.2f\n", energy);
    fflush(stdout);
    
    /* Check if blob has entrypoint (if not, skip calling it) */
    int blob_ready = (g->hdr->main_entry_offset != 0);
    if (!blob_ready) {
        printf("  [INFO] Blob is empty (main_entry_offset=0)\n");
        printf("         Will feed bytes but skip melvin_call_entry\n");
        printf("         (Blob needs to be seeded with UEL physics first)\n\n");
        fflush(stdout);
    }
    
    printf("Starting episodes...\n");
    fflush(stdout);
    
    const int bar_width = 50;
    const int update_freq = (episodes > bar_width) ? episodes / bar_width : 1;
    
    printf("Progress: [");
    for (int i = 0; i < bar_width; i++) printf(" ");
    printf("] 0/%d (0.0%%)", episodes);
    fflush(stdout);
    
    for (int ep = 0; ep < episodes; ep++) {
        /* Feed pattern bytes */
        const char *p = pattern;
        while (*p) {
            melvin_feed_byte(g, in_port, (uint8_t)*p, energy);
            p++;
        }
        
        /* Let the brain run its own laws once per episode (only if blob is ready) */
        if (blob_ready) {
            melvin_call_entry(g);
        }
        
        /* Update progress bar every episode for visibility */
        int pos = (int)((double)(ep + 1) / episodes * bar_width);
        printf("\rProgress: [");
        for (int i = 0; i < bar_width; i++) {
            if (i < pos) {
                printf("=");
            } else if (i == pos) {
                printf(">");
            } else {
                printf(" ");
            }
        }
        printf("] %d/%d (%.1f%%)", ep + 1, episodes, 
               (double)(ep + 1) / episodes * 100.0);
        fflush(stdout);
    }
    printf("\n");  /* Newline after progress bar */
    printf("Episodes complete!\n");
    fflush(stdout);
    
    printf("\n--- AFTER INPUT ---\n");
    print_node_stats(g, "Final");
    print_edge_stats(g, "Final");
    
    /* Print detailed stats */
    printf("\n--- DETAILED ANALYSIS ---\n");
    print_top_edges(g, 10);
    printf("\n");
    print_top_activations(g, 10);
    
    /* Sync and close */
    printf("\n--- SYNCING ---\n");
    melvin_sync(g);
    melvin_close(g);
    
    printf("[OK] Closed %s\n", brain_path);
    printf("\n=== Test Complete ===\n");
    printf("\nTo re-run on same brain (see if it continues learning):\n");
    printf("  ./test_first_run\n");
    printf("(Don't delete first_run.m between runs)\n");
    
    return 0;
}

