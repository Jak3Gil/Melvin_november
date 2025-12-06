/*
 * inspect_patterns.c - Diagnostic tool to inspect pattern formation
 * 
 * Checks:
 * 1. Are patterns being created?
 * 2. Are blanks being used?
 * 3. What patterns exist in the brain?
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "src/melvin.h"

#define PATTERN_MAGIC 0x4E544150  /* "PATN" */

void inspect_pattern(Graph *g, uint32_t pattern_id) {
    if (!g || pattern_id >= g->node_count) {
        printf("ERROR: Invalid pattern_id %u (node_count=%llu)\n", 
               pattern_id, (unsigned long long)g->node_count);
        return;
    }
    
    Node *n = &g->nodes[pattern_id];
    if (n->pattern_data_offset == 0) {
        printf("Node %u is not a pattern node (pattern_data_offset=0)\n", pattern_id);
        return;  /* Not a pattern node */
    }
    
    /* Read pattern data from blob */
    uint64_t pattern_offset = n->pattern_data_offset - g->hdr->blob_offset;
    if (pattern_offset >= g->blob_size) {
        printf("ERROR: Pattern offset %llu >= blob_size %llu\n",
               (unsigned long long)pattern_offset, (unsigned long long)g->blob_size);
        return;
    }
    
    PatternData *pd = (PatternData *)(g->blob + pattern_offset);
    if (pd->magic != PATTERN_MAGIC) {
        printf("ERROR: Pattern magic mismatch: got 0x%08X, expected 0x%08X\n",
               pd->magic, PATTERN_MAGIC);
        return;
    }
    
    printf("\n=== Pattern Node %u ===\n", pattern_id);
    printf("Elements: %u\n", pd->element_count);
    printf("Instances: %u\n", pd->instance_count);
    printf("Frequency: %.2f\n", pd->frequency);
    printf("Strength: %.2f\n", pd->strength);
    printf("\nElements:\n");
    
    uint32_t blank_count = 0;
    uint32_t concrete_count = 0;
    
    for (uint32_t i = 0; i < pd->element_count && i < 20; i++) {
        PatternElement *e = &pd->elements[i];
        if (e->is_blank) {
            printf("  [%u] BLANK (pos %u)\n", i, e->value);
            blank_count++;
        } else {
            uint32_t node_id = e->value;
            if (node_id < g->node_count) {
                uint8_t byte = g->nodes[node_id].byte;
                char c = (byte >= 32 && byte < 127) ? (char)byte : '?';
                printf("  [%u] CONCRETE '%c' (node %u, byte %u)\n", i, c, node_id, byte);
            } else {
                printf("  [%u] CONCRETE (node %u - invalid)\n", i, node_id);
            }
            concrete_count++;
        }
    }
    
    printf("\nSummary: %u blanks, %u concrete\n", blank_count, concrete_count);
    printf("Has blanks: %s\n", (blank_count > 0) ? "YES" : "NO");
    printf("========================\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <brain.m> [pattern_id]\n", argv[0]);
        fprintf(stderr, "  If pattern_id omitted, scans for all patterns\n");
        return 1;
    }
    
    const char *brain_path = argv[1];
    Graph *g = melvin_open(brain_path, 10000, 50000, 1048576);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to open brain: %s\n", brain_path);
        return 1;
    }
    
    printf("========================================\n");
    printf("PATTERN INSPECTION\n");
    printf("========================================\n");
    printf("Brain: %s\n", brain_path);
    printf("Node count: %llu\n", (unsigned long long)g->node_count);
    printf("Edge count: %llu\n", (unsigned long long)g->edge_count);
    printf("\n");
    
    if (argc > 2) {
        /* Inspect specific pattern */
        uint32_t pattern_id = (uint32_t)atoi(argv[2]);
        inspect_pattern(g, pattern_id);
    } else {
        /* Scan for all patterns */
        printf("Scanning for pattern nodes...\n");
        printf("(Pattern nodes typically start at node 840+)\n\n");
        
        uint32_t pattern_count = 0;
        uint32_t patterns_with_blanks = 0;
        uint32_t patterns_without_blanks = 0;
        
        for (uint64_t i = 840; i < g->node_count && i < 2000; i++) {
            if (g->nodes[i].pattern_data_offset > 0) {
                pattern_count++;
                
                /* Quick check for blanks */
                uint64_t pattern_offset = g->nodes[i].pattern_data_offset - g->hdr->blob_offset;
                if (pattern_offset < g->blob_size) {
                    PatternData *pd = (PatternData *)(g->blob + pattern_offset);
                    if (pd->magic == PATTERN_MAGIC) {
                        bool has_blanks = false;
                        for (uint32_t j = 0; j < pd->element_count && j < 20; j++) {
                            if (pd->elements[j].is_blank) {
                                has_blanks = true;
                                break;
                            }
                        }
                        
                        if (has_blanks) {
                            patterns_with_blanks++;
                        } else {
                            patterns_without_blanks++;
                        }
                    }
                }
                
                /* Show first 10 patterns in detail */
                if (pattern_count <= 10) {
                    inspect_pattern(g, (uint32_t)i);
                }
            }
        }
        
        printf("\n========================================\n");
        printf("SUMMARY\n");
        printf("========================================\n");
        printf("Total patterns found: %u\n", pattern_count);
        printf("Patterns with blanks: %u\n", patterns_with_blanks);
        printf("Patterns without blanks: %u\n", patterns_without_blanks);
        printf("Blank usage: %.1f%%\n", 
               (pattern_count > 0) ? (100.0f * patterns_with_blanks / pattern_count) : 0.0f);
        printf("\n");
        
        if (patterns_with_blanks == 0 && pattern_count > 0) {
            printf("⚠️  WARNING: No patterns have blanks!\n");
            printf("   Patterns are all concrete (too specific).\n");
            printf("   This means patterns won't generalize.\n");
        } else if (patterns_with_blanks > 0) {
            printf("✅ Good: Some patterns use blanks for generalization.\n");
        }
    }
    
    melvin_close(g);
    return 0;
}

