/*
 * Test graph-driven learning
 * Verifies that learning is triggered by LEARNER nodes, not CLI logic
 */

#include "melvin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to create graph\n");
        return 1;
    }
    
    // Create VALUE nodes for learning parameters (using hardware function directly)
    // In real usage, patterns would create these via graph_create_node()
    // For testing, we use the hardware function directly
    static uint64_t next_value_id = (1ULL << 61);
    for (int i = 0; i < 5; i++) {
        const char *names[] = {"min_pattern_len", "max_pattern_len", "output_learning_rate", "max_output_weight", "output_activation_threshold"};
        float values[] = {2.0f, 16.0f, 0.1f, 2.0f, 0.1f};
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
    
    printf("Created VALUE nodes for cognitive parameters\n");
    
    // Create data: "abc" and "xyz"
    Node *data1[3], *data2[3];
    const char *str1 = "abc";
    const char *str2 = "xyz";
    
    for (int i = 0; i < 3; i++) {
        data1[i] = graph_add_data_byte(g, str1[i]);
        data2[i] = graph_add_data_byte(g, str2[i]);
        if (i > 0) {
            graph_add_edge(g, data1[i-1]->id, data1[i]->id, 1.0f);
            graph_add_edge(g, data2[i-1]->id, data2[i]->id, 1.0f);
        }
    }
    
    // Create base patterns
    PatternAtom atoms_ab[2];
    atoms_ab[0].delta = 0;
    atoms_ab[0].mode = 0;
    atoms_ab[0].value = 'a';
    atoms_ab[1].delta = 1;
    atoms_ab[1].mode = 0;
    atoms_ab[1].value = 'b';
    
    Node *pattern_ab = graph_add_pattern(g, atoms_ab, 2, 0.5f);
    printf("Created base pattern 'ab' (id: %llu)\n", (unsigned long long)pattern_ab->id);
    
    // Create EPISODE nodes for the two data sequences (using hardware function directly)
    // In real usage, patterns would create these via graph_create_node()
    static uint64_t next_episode_id = (1ULL << 60);
    uint64_t payload1[2] = {data1[0]->id, data1[2]->id};
    uint64_t payload2[2] = {data2[0]->id, data2[2]->id};
    Node *episode1 = graph_create_node(g, NODE_EPISODE, next_episode_id++, payload1, sizeof(payload1));
    Node *episode2 = graph_create_node(g, NODE_EPISODE, next_episode_id++, payload2, sizeof(payload2));
    if (episode1) episode1->a = 1.0f;
    if (episode2) episode2->a = 1.0f;
    printf("Created EPISODE nodes: %llu, %llu\n",
           (unsigned long long)episode1->id,
           (unsigned long long)episode2->id);
    
    // Build explanations and convert to graph
    Explanation exp1, exp2;
    explanation_init(&exp1);
    explanation_init(&exp2);
    
    graph_build_explanation_single_pattern(g, pattern_ab, data1[0]->id, data1[2]->id, 0.9f, &exp1);
    graph_build_explanation_single_pattern(g, pattern_ab, data2[0]->id, data2[2]->id, 0.9f, &exp2);
    
    explanation_to_graph(g, &exp1, episode1->id);
    explanation_to_graph(g, &exp2, episode2->id);
    
    printf("Converted explanations to APPLICATION nodes\n");
    
    // Create LEARNER node and connect to episodes (using hardware function directly)
    // In real usage, patterns would create these via graph_create_node()
    static uint64_t next_learner_id = (1ULL << 58);
    Node *learner = graph_create_node(g, NODE_LEARNER, next_learner_id++, NULL, 0);
    if (learner) {
        learner->a = 1.0f;  // Activate learner
        
        // Connect LEARNER -> EPISODE
        graph_add_edge(g, learner->id, episode1->id, 1.0f);
        graph_add_edge(g, learner->id, episode2->id, 1.0f);
        
        printf("Created LEARNER node %llu connected to episodes\n", (unsigned long long)learner->id);
    }
    
    // Count patterns before
    uint64_t patterns_before = 0;
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].kind == NODE_PATTERN) patterns_before++;
    }
    printf("Patterns before learning: %llu\n", (unsigned long long)patterns_before);
    
    // Run graph-driven learning (should create new patterns from episodes)
    graph_run_local_rules(g);
    
    // Count patterns after
    uint64_t patterns_after = 0;
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].kind == NODE_PATTERN) patterns_after++;
    }
    printf("Patterns after learning: %llu\n", (unsigned long long)patterns_after);
    
    if (patterns_after > patterns_before) {
        printf("✓ Graph-driven learning created new patterns\n");
    } else {
        printf("⚠ No new patterns created (may need more episodes or different data)\n");
    }
    
    // Test VALUE node lookup
    float min_len = graph_get_value(g, "min_pattern_len", 999.0f);
    float max_len = graph_get_value(g, "max_pattern_len", 999.0f);
    printf("VALUE nodes: min_pattern_len=%.1f, max_pattern_len=%.1f\n", min_len, max_len);
    
    if (min_len == 2.0f && max_len == 16.0f) {
        printf("✓ VALUE nodes work correctly\n");
    } else {
        printf("✗ VALUE node lookup failed\n");
    }
    
    explanation_free(&exp1);
    explanation_free(&exp2);
    graph_destroy(g);
    
    return (patterns_after > patterns_before && min_len == 2.0f) ? 0 : 1;
}

