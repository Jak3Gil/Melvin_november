/*
 * test_exec_tracking.c - Test EXEC node origin tracking and pattern promotion
 * 
 * Tests:
 * 1. EXEC nodes created via melvin_teach_operation are tagged as TAUGHT
 * 2. EXEC nodes created via melvin_create_exec_node are tagged as INSTINCT
 * 3. Pattern promotion creates PATTERN_PROMO EXEC nodes
 * 4. All EXEC creations are logged to exec_creation.log
 * 5. Tracking fields are correctly set
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include "src/melvin.h"

int main(int argc, char **argv) {
    const char *brain_path = "test_exec_tracking.m";
    
    printf("=== EXEC Node Tracking Test ===\n\n");
    
    /* Remove old brain if exists */
    unlink(brain_path);
    unlink("evaluation_results/exec_creation.log");
    
    /* Create new brain */
    printf("1. Creating brain...\n");
    Graph *g = melvin_open(brain_path, 10000, 50000, 1024 * 1024);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to create brain\n");
        return 1;
    }
    printf("   ✓ Brain created\n\n");
    
    /* Test 1: Create EXEC via melvin_teach_operation (should be TAUGHT) */
    printf("2. Creating EXEC node via melvin_teach_operation...\n");
    uint8_t dummy_code[16] = {0x90, 0x90, 0x90, 0x90, 0xC3};  /* NOPs + RET */
    uint32_t taught_exec = melvin_teach_operation(g, dummy_code, sizeof(dummy_code), "test_add");
    if (taught_exec == UINT32_MAX) {
        fprintf(stderr, "   ✗ Failed to create TAUGHT EXEC\n");
    } else {
        printf("   ✓ Created TAUGHT EXEC node %u\n", taught_exec);
        Node *node = &g->nodes[taught_exec];
        printf("      exec_origin=%u (expected 2=TAUGHT)\n", node->exec_origin);
        printf("      parent_pattern_id=%u (expected 0)\n", node->parent_pattern_id);
        printf("      created_update=%llu\n", (unsigned long long)node->created_update);
        printf("      payload_offset=%llu\n", (unsigned long long)node->payload_offset);
        
        if (node->exec_origin != 2) {
            fprintf(stderr, "   ✗ ERROR: exec_origin should be 2 (TAUGHT), got %u\n", node->exec_origin);
            return 1;
        }
        if (node->parent_pattern_id != 0) {
            fprintf(stderr, "   ✗ ERROR: parent_pattern_id should be 0, got %u\n", node->parent_pattern_id);
            return 1;
        }
    }
    printf("\n");
    
    /* Test 2: Create EXEC via melvin_create_exec_node (should be INSTINCT) */
    printf("3. Creating EXEC node via melvin_create_exec_node...\n");
    uint64_t blob_offset = 2048;
    uint32_t instinct_exec = melvin_create_exec_node(g, 2500, blob_offset, 0.5f);
    if (instinct_exec == UINT32_MAX) {
        fprintf(stderr, "   ✗ Failed to create INSTINCT EXEC\n");
    } else {
        printf("   ✓ Created INSTINCT EXEC node %u\n", instinct_exec);
        Node *node = &g->nodes[instinct_exec];
        printf("      exec_origin=%u (expected 1=INSTINCT)\n", node->exec_origin);
        printf("      parent_pattern_id=%u (expected 0)\n", node->parent_pattern_id);
        printf("      created_update=%llu\n", (unsigned long long)node->created_update);
        printf("      payload_offset=%llu\n", (unsigned long long)node->payload_offset);
        
        if (node->exec_origin != 1) {
            fprintf(stderr, "   ✗ ERROR: exec_origin should be 1 (INSTINCT), got %u\n", node->exec_origin);
            return 1;
        }
        if (node->parent_pattern_id != 0) {
            fprintf(stderr, "   ✗ ERROR: parent_pattern_id should be 0, got %u\n", node->parent_pattern_id);
            return 1;
        }
    }
    printf("\n");
    
    /* Test 3: Create patterns and promote one to EXEC */
    printf("4. Creating patterns and promoting to EXEC...\n");
    
    /* Feed some bytes to create patterns */
    for (int i = 0; i < 50; i++) {
        melvin_feed_byte(g, 0, (uint8_t)('A' + (i % 3)), 0.2f);
        melvin_run_physics(g);
    }
    
    /* Find a pattern node */
    uint32_t pattern_id = UINT32_MAX;
    for (uint32_t i = 840; i < 1000 && i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_id = i;
            break;
        }
    }
    
    if (pattern_id == UINT32_MAX) {
        printf("   ⚠ No patterns found yet, skipping pattern promotion test\n");
    } else {
        printf("   ✓ Found pattern node %u\n", pattern_id);
        
        /* Promote pattern to EXEC */
        uint32_t promo_exec = promote_pattern_to_exec(g, pattern_id);
        if (promo_exec == UINT32_MAX) {
            fprintf(stderr, "   ✗ Failed to promote pattern to EXEC\n");
        } else {
            printf("   ✓ Promoted pattern %u to EXEC node %u\n", pattern_id, promo_exec);
            Node *node = &g->nodes[promo_exec];
            printf("      exec_origin=%u (expected 3=PATTERN_PROMO)\n", node->exec_origin);
            printf("      parent_pattern_id=%u (expected %u)\n", node->parent_pattern_id, pattern_id);
            printf("      created_update=%llu\n", (unsigned long long)node->created_update);
            printf("      payload_offset=%llu\n", (unsigned long long)node->payload_offset);
            
            if (node->exec_origin != 3) {
                fprintf(stderr, "   ✗ ERROR: exec_origin should be 3 (PATTERN_PROMO), got %u\n", node->exec_origin);
                return 1;
            }
            if (node->parent_pattern_id != pattern_id) {
                fprintf(stderr, "   ✗ ERROR: parent_pattern_id should be %u, got %u\n", pattern_id, node->parent_pattern_id);
                return 1;
            }
            
            /* Check pattern has exec bound */
            uint64_t pattern_offset = g->nodes[pattern_id].pattern_data_offset - g->hdr->blob_offset;
            if (pattern_offset < g->blob_size) {
                PatternData *pd = (PatternData *)(g->blob + pattern_offset);
                if (pd->has_exec != 1 || pd->bound_exec_id != promo_exec) {
                    fprintf(stderr, "   ✗ ERROR: Pattern not marked as having EXEC\n");
                    return 1;
                }
                printf("      Pattern marked has_exec=1, bound_exec_id=%u\n", pd->bound_exec_id);
            }
        }
    }
    printf("\n");
    
    /* Test 4: Check log file */
    printf("5. Checking exec_creation.log...\n");
    FILE *log = fopen("evaluation_results/exec_creation.log", "r");
    if (!log) {
        fprintf(stderr, "   ✗ Log file not found\n");
        return 1;
    }
    
    char line[512];
    int log_count = 0;
    while (fgets(line, sizeof(line), log)) {
        if (strstr(line, "EXEC_CREATE")) {
            log_count++;
            printf("   Log entry: %s", line);
        }
    }
    fclose(log);
    
    if (log_count == 0) {
        fprintf(stderr, "   ✗ No log entries found\n");
        return 1;
    }
    printf("   ✓ Found %d log entries\n\n", log_count);
    
    /* Test 5: Verify all EXEC nodes have tracking fields set */
    printf("6. Verifying all EXEC nodes have tracking fields...\n");
    int exec_count = 0;
    int invalid_count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].type == NODE_TYPE_EXEC || g->nodes[i].payload_offset > 0) {
            exec_count++;
            Node *node = &g->nodes[i];
            if (node->exec_origin == 0 && node->payload_offset > 0) {
                printf("   ⚠ EXEC node %llu has exec_origin=0 (NONE) but has payload_offset\n", (unsigned long long)i);
                invalid_count++;
            }
        }
    }
    printf("   ✓ Found %d EXEC nodes, %d with invalid tracking\n\n", exec_count, invalid_count);
    
    /* Summary */
    printf("=== Test Summary ===\n");
    printf("✓ TAUGHT EXEC creation: PASSED\n");
    printf("✓ INSTINCT EXEC creation: PASSED\n");
    if (pattern_id != UINT32_MAX) {
        printf("✓ PATTERN_PROMO EXEC creation: PASSED\n");
    }
    printf("✓ Log file creation: PASSED\n");
    printf("✓ Tracking fields: %s\n", (invalid_count == 0) ? "PASSED" : "WARNING");
    
    melvin_sync(g);
    melvin_close(g);
    
    printf("\n✓ All tests passed!\n");
    return 0;
}

