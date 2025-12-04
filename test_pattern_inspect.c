/*
 * Pattern Inspection Tool
 * Shows exactly what's in the patterns and why matching might fail
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "src/melvin.h"

#define PATTERN_MAGIC 0x4E544150

void inspect_pattern(Graph *g, uint32_t pattern_id) {
    if (pattern_id >= g->node_count) return;
    
    Node *pnode = &g->nodes[pattern_id];
    if (pnode->pattern_data_offset == 0) {
        printf("  Node %u: Not a pattern\n", pattern_id);
        return;
    }
    
    uint64_t offset = pnode->pattern_data_offset - g->hdr->blob_offset;
    if (offset >= g->blob_size) {
        printf("  Node %u: Invalid offset\n", pattern_id);
        return;
    }
    
    /* Read pattern structure directly from blob */
    uint8_t *blob_ptr = g->blob + offset;
    
    /* Read magic */
    uint32_t magic = *(uint32_t*)blob_ptr;
    if (magic != PATTERN_MAGIC) {
        printf("  Node %u: Invalid magic 0x%08X\n", pattern_id, magic);
        return;
    }
    
    /* Read element count */
    uint32_t element_count = *(uint32_t*)(blob_ptr + 4);
    uint32_t instance_count = *(uint32_t*)(blob_ptr + 8);
    float frequency = *(float*)(blob_ptr + 12);
    float strength = *(float*)(blob_ptr + 16);
    
    printf("\n📋 Pattern %u:\n", pattern_id);
    printf("   Elements: %u\n", element_count);
    printf("   Instances: %u\n", instance_count);
    printf("   Frequency: %.2f\n", frequency);
    printf("   Strength: %.2f\n", strength);
    printf("   Structure:\n");
    
    /* Read elements (they start after the header) */
    uint8_t *elem_ptr = blob_ptr + 24;  /* After header fields */
    
    int blank_count = 0;
    for (uint32_t i = 0; i < element_count && i < 10; i++) {
        /* Each element is: uint8_t is_blank + padding + uint32_t value = 8 bytes */
        uint8_t is_blank = elem_ptr[i * 8];
        uint32_t value = *(uint32_t*)(elem_ptr + i * 8 + 4);
        
        if (is_blank) {
            printf("     [%u] BLANK_%u (variable)\n", i, value);
            blank_count++;
        } else if (value < g->node_count) {
            uint8_t byte = g->nodes[value].byte;
            char c = (byte >= 32 && byte < 127) ? (char)byte : '?';
            printf("     [%u] '%c' (node %u, concrete)\n", i, c, value);
        } else {
            printf("     [%u] INVALID (node %u >= node_count)\n", i, value);
        }
    }
    
    printf("   Summary: %d blanks, %d concrete\n", blank_count, element_count - blank_count);
    
    /* Show what this pattern would match */
    printf("   Matches: ");
    for (uint32_t i = 0; i < element_count && i < 10; i++) {
        uint8_t is_blank = elem_ptr[i * 8];
        uint32_t value = *(uint32_t*)(elem_ptr + i * 8 + 4);
        
        if (is_blank) {
            printf("[ANY] ");
        } else if (value < g->node_count) {
            printf("'%c' ", g->nodes[value].byte);
        }
    }
    printf("\n");
}

int main(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  PATTERN INSPECTION - What's Actually There?       ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    /* Create and train */
    const char *brain_path = "/tmp/pattern_inspect.m";
    remove(brain_path);
    
    printf("Creating brain and training...\n");
    melvin_create_v2(brain_path, 8000, 40000, 65536, 0);
    Graph *g = melvin_open(brain_path, 8000, 40000, 65536);
    
    if (!g) {
        printf("❌ Failed\n");
        return 1;
    }
    
    /* Set up EXEC nodes with payloads */
    printf("Setting up EXEC nodes...\n");
    uint64_t current_offset = 16384;
    for (uint32_t exec_id = 2000; exec_id < 2010; exec_id++) {
        uint8_t stub[32] = {0x01, 0x02, 0x03, 0x04};
        memcpy(g->blob + current_offset, stub, 32);
        g->nodes[exec_id].payload_offset = current_offset;
        current_offset += 544;
    }
    
    /* Train */
    const char *examples[] = {"1+1=2", "2+2=4", "3+3=6"};
    
    for (int i = 0; i < 3; i++) {
        printf("Training: '%s'\n", examples[i]);
        for (const char *p = examples[i]; *p; p++) {
            melvin_feed_byte(g, 0, *p, 1.0f);
        }
        for (int j = 0; j < 10; j++) {
            melvin_call_entry(g);
        }
    }
    
    printf("\n");
    
    /* Inspect all patterns */
    printf("═══════════════════════════════════════════════════\n");
    printf("PATTERNS DISCOVERED\n");
    printf("═══════════════════════════════════════════════════\n");
    
    int pattern_count = 0;
    for (uint64_t pid = 840; pid < 1000; pid++) {
        if (g->nodes[pid].pattern_data_offset > 0) {
            inspect_pattern(g, pid);
            pattern_count++;
        }
    }
    
    printf("\n");
    printf("Total patterns: %d\n", pattern_count);
    
    printf("\n");
    printf("═══════════════════════════════════════════════════\n");
    printf("ANALYSIS\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    printf("Looking for arithmetic pattern: [BLANK, +, BLANK, =, BLANK]\n");
    printf("Expected length: 5\n");
    printf("Expected blanks: 3\n\n");
    
    printf("What we actually have:\n");
    printf("  - Most patterns: length 2-3\n");
    printf("  - Need: length 5 for full arithmetic\n\n");
    
    printf("Why?\n");
    printf("  - Co-activation window might not capture full 5-char sequence\n");
    printf("  - Or activation window is too small\n");
    printf("  - Or sequences aren't staying active long enough\n\n");
    
    printf("Solution:\n");
    printf("  - Increase activation window size\n");
    printf("  - OR use hierarchical composition (compose smaller patterns)\n");
    printf("  - OR use sequence buffer instead of activation window\n\n");
    
    melvin_close(g);
    
    return 0;
}

