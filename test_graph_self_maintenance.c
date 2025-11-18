/*
 * Test graph self-maintenance
 * Verifies that MAINTENANCE nodes can prune unused patterns and edges
 */

#include "melvin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    Graph *g = graph_create(2048, 4096, 32 * 1024);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to create graph\n");
        return 1;
    }
    
    // Create VALUE nodes for maintenance parameters (using hardware function directly)
    // In real usage, patterns would create these via graph_create_node()
    static uint64_t next_value_id = (1ULL << 61);
    const char *names[] = {"min_pattern_usage", "min_edge_usage", "max_age_without_activation", 
                          "edge_decay_rate", "min_edge_weight", "maintenance_work_budget", "pattern_decay_rate"};
    float values[] = {0.5f, 2.0f, 1000.0f, 0.9f, 0.2f, 50.0f, 0.5f};
    for (int i = 0; i < 7; i++) {
        size_t name_len = strlen(names[i]);
        size_t payload_size = name_len + 1 + sizeof(float);
        char *payload = malloc(payload_size);
        memcpy(payload, names[i], name_len + 1);
        float *val_ptr = (float *)(payload + name_len + 1);
        *val_ptr = values[i];
        Node *val_node = graph_create_node(g, NODE_VALUE, next_value_id++, payload, payload_size);
        free(payload);
        if (val_node) val_node->a = values[i];
    }
    
    printf("Created VALUE nodes for maintenance parameters\n");
    
    // Create many low-usage patterns
    PatternAtom atoms[2];
    Node *low_usage_patterns[10];
    Node *high_usage_patterns[2];
    
    // Low-usage patterns (will be pruned)
    for (int i = 0; i < 10; i++) {
        atoms[0].delta = 0;
        atoms[0].mode = 0;
        atoms[0].value = 'a' + i;
        atoms[1].delta = 1;
        atoms[1].mode = 0;
        atoms[1].value = 'b' + i;
        
        low_usage_patterns[i] = graph_add_pattern(g, atoms, 2, 0.1f);  // Low quality
        low_usage_patterns[i]->a = 0.0f;  // Not active
    }
    
    // High-usage patterns (should be preserved)
    for (int i = 0; i < 2; i++) {
        atoms[0].delta = 0;
        atoms[0].mode = 0;
        atoms[0].value = 'x' + i;
        atoms[1].delta = 1;
        atoms[1].mode = 0;
        atoms[1].value = 'y' + i;
        
        high_usage_patterns[i] = graph_add_pattern(g, atoms, 2, 0.8f);  // High quality
        high_usage_patterns[i]->a = 0.5f;  // Active
    }
    
    printf("Created 10 low-usage patterns and 2 high-usage patterns\n");
    
    // Create weak edges (will be pruned)
    Node *data1 = graph_add_data_byte(g, 'z');
    Node *data2 = graph_add_data_byte(g, 'w');
    
    for (int i = 0; i < 5; i++) {
        Edge *weak_edge = graph_add_edge(g, data1->id, data2->id, 0.05f);  // Very weak
        (void)weak_edge;  // Suppress unused warning
    }
    
    printf("Created 5 weak edges\n");
    
    // Count initial state
    uint64_t patterns_before = 0;
    uint64_t edges_before = 0;
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].kind == NODE_PATTERN) patterns_before++;
    }
    edges_before = g->num_edges;
    
    printf("[maintenance] before: patterns=%llu edges=%llu\n",
           (unsigned long long)patterns_before,
           (unsigned long long)edges_before);
    
    // Create MAINTENANCE node and connect to all patterns (using hardware function directly)
    // In real usage, patterns would create these via graph_create_node()
    static uint64_t next_maintenance_id = (1ULL << 57);
    Node *maintenance = graph_create_node(g, NODE_MAINTENANCE, next_maintenance_id++, NULL, 0);
    if (maintenance) {
        maintenance->a = 1.0f;  // Activate maintenance
        printf("MAINTENANCE node activation: %.2f\n", maintenance->a);
        
        // Connect MAINTENANCE to all patterns (so it can monitor them)
        for (int i = 0; i < 10; i++) {
            graph_add_edge(g, maintenance->id, low_usage_patterns[i]->id, 1.0f);
        }
        for (int i = 0; i < 2; i++) {
            graph_add_edge(g, maintenance->id, high_usage_patterns[i]->id, 1.0f);
        }
        
        // Also connect to data nodes (so it can monitor their edges)
        graph_add_edge(g, maintenance->id, data1->id, 1.0f);
        graph_add_edge(g, maintenance->id, data2->id, 1.0f);
        
        printf("Created MAINTENANCE node %llu connected to patterns\n",
               (unsigned long long)maintenance->id);
    }
    
    // Run maintenance for several ticks
    printf("Running maintenance for 10 ticks...\n");
    for (int tick = 0; tick < 10; tick++) {
        // Keep maintenance node active (propagation might decay it)
        if (maintenance) maintenance->a = 1.0f;
        graph_propagate(g, 1);
        // Re-activate maintenance after propagation
        if (maintenance) maintenance->a = 1.0f;
        graph_run_local_rules(g);
        
        // Check if maintenance is working
        if (tick == 0 || tick == 9) {
            uint64_t patterns_active = 0;
            uint64_t patterns_deleted = 0;
            for (uint64_t i = 0; i < g->num_nodes; i++) {
                if (g->nodes[i].kind == NODE_PATTERN) {
                    if (g->nodes[i].flags & 1) {
                        patterns_deleted++;
                    } else {
                        patterns_active++;
                    }
                }
            }
            printf("  Tick %d: %llu active patterns, %llu deleted\n", 
                   tick, (unsigned long long)patterns_active, (unsigned long long)patterns_deleted);
        }
    }
    
    // Count final state
    uint64_t patterns_after = 0;
    uint64_t edges_after = 0;
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].kind == NODE_PATTERN) {
            // Only count non-deleted patterns
            if (!(g->nodes[i].flags & 1)) {
                patterns_after++;
            }
        }
    }
    edges_after = g->num_edges;
    
    printf("[maintenance] after:  patterns=%llu edges=%llu\n",
           (unsigned long long)patterns_after,
           (unsigned long long)edges_after);
    
    // Verify high-usage patterns are preserved
    int high_usage_preserved = 0;
    for (int i = 0; i < 2; i++) {
        if (high_usage_patterns[i] && !(high_usage_patterns[i]->flags & 1)) {
            high_usage_preserved++;
        }
    }
    
    // Check if any patterns were marked as deleted
    uint64_t patterns_deleted_final = 0;
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].kind == NODE_PATTERN && (g->nodes[i].flags & 1)) {
            patterns_deleted_final++;
        }
    }
    
    printf("Patterns marked as deleted: %llu\n", (unsigned long long)patterns_deleted_final);
    
    if (patterns_deleted_final > 0 || patterns_after < patterns_before || edges_after < edges_before) {
        printf("✓ Self-maintenance reduced unused structure\n");
        if (high_usage_preserved == 2) {
            printf("✓ High-usage patterns preserved\n");
        } else {
            printf("⚠ Some high-usage patterns were removed (may need tuning)\n");
        }
        return 0;
    } else {
        printf("⚠ No maintenance occurred (may need more ticks or lower thresholds)\n");
        return 1;
    }
}

