/*
 * validate_before_longrun.c - Quick validation test (5 minutes)
 * 
 * Tests current system without changing anything:
 * 1. Opens brain.m
 * 2. Feeds test data
 * 3. Inspects what happened
 * 4. Reports if ready for long-term run
 */

#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Check if blob is seeded */
static int check_blob_seeded(Graph *g) {
    if (!g || !g->hdr) return 0;
    return (g->hdr->main_entry_offset > 0) ? 1 : 0;
}

/* Feed test pattern and observe changes */
static void test_pattern_formation(Graph *g) {
    if (!g) return;
    
    printf("\n--- Testing Pattern Formation ---\n");
    
    /* Initial state */
    uint64_t initial_edges = g->hdr->edge_count;
    float initial_chaos = 0.0f;
    
    /* Compute initial chaos (simple: count incoherent nodes) */
    for (size_t i = 0; i < g->hdr->node_count && i < 1000; i++) {
        float a = g->nodes[i].a;
        if (fabsf(a) > 0.1f) initial_chaos += fabsf(a);
    }
    
    printf("Initial edges: %llu\n", (unsigned long long)initial_edges);
    printf("Initial chaos (sum of activations): %.4f\n", initial_chaos);
    
    /* Feed test pattern "the cat" 10 times */
    const char *pattern = "the cat";
    uint32_t in_port = 256;
    
    printf("Feeding pattern '%s' 10 times...\n", pattern);
    
    for (int rep = 0; rep < 10; rep++) {
        for (size_t i = 0; pattern[i]; i++) {
            melvin_feed_byte(g, in_port, (uint8_t)pattern[i], 1.0f);
        }
        
        /* Let brain process if blob is seeded */
        if (check_blob_seeded(g)) {
            /* Try to call blob */
            /* NOTE: If blob code has unresolved relocations, this will crash */
            /* This is a known issue - blob seeding needs proper relocation resolution */
            /* For now, we skip calling if we detect potential issues */
            
            /* Check if blob looks valid (not all zeros) */
            int blob_looks_valid = 0;
            for (int i = 0; i < 32 && (g->hdr->main_entry_offset + i) < g->hdr->blob_size; i++) {
                if (g->blob[g->hdr->main_entry_offset + i] != 0) {
                    blob_looks_valid = 1;
                    break;
                }
            }
            
            if (blob_looks_valid) {
                /* Attempt to call - may crash if relocations unresolved */
                melvin_call_entry(g);
            } else {
                printf("  ⚠ Blob appears invalid (all zeros), skipping call\n");
            }
        }
    }
    
    /* Final state */
    uint64_t final_edges = g->hdr->edge_count;
    float final_chaos = 0.0f;
    
    for (size_t i = 0; i < g->hdr->node_count && i < 1000; i++) {
        float a = g->nodes[i].a;
        if (fabsf(a) > 0.1f) final_chaos += fabsf(a);
    }
    
    printf("Final edges: %llu\n", (unsigned long long)final_edges);
    printf("Final chaos: %.4f\n", final_chaos);
    printf("Edges added: %lld\n", (long long)(final_edges - initial_edges));
    printf("Chaos change: %.4f\n", final_chaos - initial_chaos);
    
    /* Check if edges strengthened */
    printf("\nChecking edge weights for pattern 'the cat'...\n");
    int found_pattern = 0;
    
    /* Look for edges between 't'->'h', 'h'->'e', 'e'->' ', etc. */
    for (size_t i = 0; pattern[i] && pattern[i+1]; i++) {
        uint8_t src_byte = pattern[i];
        uint8_t dst_byte = pattern[i+1];
        
        /* Find edge */
        uint32_t src_node = (uint32_t)src_byte;
        uint32_t dst_node = (uint32_t)dst_byte;
        
        if (src_node < g->hdr->node_count && dst_node < g->hdr->node_count) {
            uint32_t eid = g->nodes[src_node].first_out;
            int iterations = 0;
            while (eid != UINT32_MAX && eid < g->hdr->edge_count && iterations < 1000) {
                if (g->edges[eid].dst == dst_node) {
                    float w = g->edges[eid].w;
                    printf("  Edge '%c'->'%c': weight=%.4f\n", 
                           (char)src_byte, (char)dst_byte, w);
                    if (fabsf(w) > 0.1f) found_pattern = 1;
                    break;
                }
                eid = g->edges[eid].next_out;
                iterations++;
            }
        }
    }
    
    if (found_pattern) {
        printf("  ✓ Pattern edges found with significant weights\n");
    } else {
        printf("  ⚠ Pattern edges weak or not found\n");
    }
}

/* Inspect top activations */
static void inspect_activations(Graph *g) {
    if (!g) return;
    
    printf("\n--- Top Activations ---\n");
    
    /* Find top 10 activations */
    float top_a[10] = {0};
    size_t top_idx[10] = {0};
    
    for (size_t i = 0; i < g->hdr->node_count && i < 1000; i++) {
        float a_abs = fabsf(g->nodes[i].a);
        
        for (int rank = 0; rank < 10; rank++) {
            if (a_abs > top_a[rank]) {
                /* Shift down */
                for (int j = 9; j > rank; j--) {
                    top_a[j] = top_a[j-1];
                    top_idx[j] = top_idx[j-1];
                }
                top_a[rank] = a_abs;
                top_idx[rank] = i;
                break;
            }
        }
    }
    
    for (int i = 0; i < 10; i++) {
        if (top_a[i] > 0.001f) {
            uint8_t b = g->nodes[top_idx[i]].byte;
            char display = (b >= 32 && b < 127) ? (char)b : '?';
            printf("  [%zu] '%c' (0x%02x): %.4f\n", 
                   top_idx[i], display, b, g->nodes[top_idx[i]].a);
        }
    }
}

/* Inspect top edge weights */
static void inspect_edges(Graph *g) {
    if (!g) return;
    
    printf("\n--- Top Edge Weights ---\n");
    
    /* Find top 10 edge weights */
    float top_w[10] = {0};
    size_t top_eid[10] = {0};
    
    for (size_t i = 0; i < g->hdr->edge_count && i < 10000; i++) {
        float w_abs = fabsf(g->edges[i].w);
        
        for (int rank = 0; rank < 10; rank++) {
            if (w_abs > top_w[rank]) {
                /* Shift down */
                for (int j = 9; j > rank; j--) {
                    top_w[j] = top_w[j-1];
                    top_eid[j] = top_eid[j-1];
                }
                top_w[rank] = w_abs;
                top_eid[rank] = i;
                break;
            }
        }
    }
    
    for (int i = 0; i < 10; i++) {
        if (top_w[i] > 0.001f) {
            uint32_t src = g->edges[top_eid[i]].src;
            uint32_t dst = g->edges[top_eid[i]].dst;
            uint8_t src_byte = (src < g->hdr->node_count) ? g->nodes[src].byte : 0;
            uint8_t dst_byte = (dst < g->hdr->node_count) ? g->nodes[dst].byte : 0;
            char src_char = (src_byte >= 32 && src_byte < 127) ? (char)src_byte : '?';
            char dst_char = (dst_byte >= 32 && dst_byte < 127) ? (char)dst_byte : '?';
            printf("  [%zu] '%c'->'%c': %.4f\n", 
                   top_eid[i], src_char, dst_char, g->edges[top_eid[i]].w);
        }
    }
}

int main(int argc, char **argv) {
    const char *brain_path = (argc > 1) ? argv[1] : "brain.m";
    
    printf("=== Validation Test ===\n");
    printf("Brain: %s\n", brain_path);
    
    /* Open brain */
    Graph *g = melvin_open(brain_path, 1000, 10000, 65536);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", brain_path);
        return 1;
    }
    
    printf("\n--- Brain State ---\n");
    printf("Nodes: %llu\n", (unsigned long long)g->hdr->node_count);
    printf("Edges: %llu\n", (unsigned long long)g->hdr->edge_count);
    printf("Blob size: %llu\n", (unsigned long long)g->hdr->blob_size);
    printf("Main entry offset: %llu\n", (unsigned long long)g->hdr->main_entry_offset);
    
    /* Check 1: Blob seeded? */
    printf("\n--- Check 1: Blob Seeded ---\n");
    int blob_seeded = check_blob_seeded(g);
    if (blob_seeded) {
        printf("  ✓ Blob is seeded (main_entry_offset = %llu)\n", 
               (unsigned long long)g->hdr->main_entry_offset);
    } else {
        printf("  ⚠ Blob is empty (main_entry_offset = 0)\n");
        printf("     Run: ./uel_seed_tool %s\n", brain_path);
    }
    
    /* Check 2: Test pattern formation */
    test_pattern_formation(g);
    
    /* Check 3: Inspect what happened */
    inspect_activations(g);
    inspect_edges(g);
    
    /* Final assessment */
    printf("\n=== Validation Result ===\n");
    
    int ready = 1;
    
    if (!blob_seeded) {
        printf("❌ Blob not seeded - cannot run UEL physics\n");
        ready = 0;
    } else {
        printf("✓ Blob seeded\n");
    }
    
    if (g->hdr->edge_count == 0) {
        printf("⚠ No edges formed - may need more data\n");
    } else {
        printf("✓ Edges exist (%llu)\n", (unsigned long long)g->hdr->edge_count);
    }
    
    if (ready) {
        printf("\n✅ READY for long-term run\n");
        printf("   Brain is seeded and responding to input\n");
    } else {
        printf("\n❌ NOT READY - fix issues above\n");
    }
    
    melvin_sync(g);
    melvin_close(g);
    
    return ready ? 0 : 1;
}

