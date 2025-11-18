/*
 * Test pattern-on-pattern learning
 * Verifies that build_symbol_sequence_from_episode returns PATTERN node IDs
 * when patterns match, and that graph_create_pattern_from_sequences works
 */

#include "melvin.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to create graph\n");
        return 1;
    }
    
    // Create data: "abcabc"
    Node *data_nodes[6];
    for (int i = 0; i < 6; i++) {
        uint8_t byte = (i < 3) ? ('a' + i) : ('a' + (i - 3));
        data_nodes[i] = graph_add_data_byte(g, byte);
        if (i > 0) {
            graph_add_edge(g, data_nodes[i-1]->id, data_nodes[i]->id, 1.0f);
        }
    }
    
    uint64_t start_id = data_nodes[0]->id;
    uint64_t end_id = data_nodes[5]->id;
    
    // Create a pattern for "ab"
    PatternAtom atoms_ab[2];
    atoms_ab[0].delta = 0;
    atoms_ab[0].mode = 0;
    atoms_ab[0].value = 'a';
    atoms_ab[1].delta = 1;
    atoms_ab[1].mode = 0;
    atoms_ab[1].value = 'b';
    
    Node *pattern_ab = graph_add_pattern(g, atoms_ab, 2, 0.5f);
    if (!pattern_ab) {
        fprintf(stderr, "ERROR: Failed to create pattern 'ab'\n");
        graph_destroy(g);
        return 1;
    }
    
    // Build explanation for the data span
    Explanation exp;
    explanation_init(&exp);
    graph_build_explanation_single_pattern(g, pattern_ab, start_id, end_id, 0.9f, &exp);
    
    printf("Created pattern 'ab' (id: %llu)\n", (unsigned long long)pattern_ab->id);
    printf("Explanation has %zu applications\n", exp.count);
    
    // Build symbol sequence from episode
    uint64_t symbol_seq[64];
    uint16_t seq_len = 0;
    
    build_symbol_sequence_from_episode(g, &exp, start_id, end_id, symbol_seq, &seq_len, 64);
    
    printf("Symbol sequence length: %u\n", seq_len);
    printf("Symbol sequence: ");
    for (uint16_t i = 0; i < seq_len; i++) {
        Node *node = graph_find_node_by_id(g, symbol_seq[i]);
        if (node && node->kind == NODE_PATTERN) {
            printf("P%llu ", (unsigned long long)node->id);
        } else {
            printf("D%llu ", (unsigned long long)symbol_seq[i]);
        }
    }
    printf("\n");
    
    // Check if pattern ID appears in sequence
    int found_pattern = 0;
    for (uint16_t i = 0; i < seq_len; i++) {
        if (symbol_seq[i] == pattern_ab->id) {
            found_pattern = 1;
            break;
        }
    }
    
    if (found_pattern) {
        printf("✓ Pattern node ID found in symbol sequence\n");
    } else {
        printf("✗ Pattern node ID NOT found in symbol sequence\n");
    }
    
    // Test pattern creation from two sequences
    // Create second data: "abxab" (different middle)
    Graph *g2 = graph_create(1024, 2048, 16 * 1024);
    Node *data2[5];
    const char *data2_str = "abxab";
    for (int i = 0; i < 5; i++) {
        data2[i] = graph_add_data_byte(g2, data2_str[i]);
        if (i > 0) {
            graph_add_edge(g2, data2[i-1]->id, data2[i]->id, 1.0f);
        }
    }
    
    // Create pattern for "ab" in second graph
    Node *pattern_ab2 = graph_add_pattern(g2, atoms_ab, 2, 0.5f);
    Explanation exp2;
    explanation_init(&exp2);
    graph_build_explanation_single_pattern(g2, pattern_ab2, data2[0]->id, data2[4]->id, 0.9f, &exp2);
    
    uint64_t symbol_seq2[64];
    uint16_t seq_len2 = 0;
    build_symbol_sequence_from_episode(g2, &exp2, data2[0]->id, data2[4]->id, symbol_seq2, &seq_len2, 64);
    
    printf("\nSecond symbol sequence length: %u\n", seq_len2);
    
    // Create pattern from two sequences (using first graph)
    // Note: sequences must reference nodes in the same graph
    // For this test, we'll create simple sequences manually
    uint64_t seq1[3] = {data_nodes[0]->id, data_nodes[1]->id, data_nodes[2]->id};
    uint64_t seq2[3] = {data_nodes[3]->id, data_nodes[4]->id, data_nodes[5]->id};
    
    Node *new_pattern = graph_create_pattern_from_sequences(g, seq1, 3, seq2, 3);
    if (new_pattern) {
        printf("✓ Created pattern from two sequences (id: %llu)\n", (unsigned long long)new_pattern->id);
        
        // Check if pattern has blanks (since sequences are identical, it shouldn't)
        size_t num_atoms = new_pattern->payload_len / sizeof(PatternAtom);
        const PatternAtom *atoms = (const PatternAtom *)(g->blob + new_pattern->payload_offset);
        int has_blanks = 0;
        for (size_t i = 0; i < num_atoms; i++) {
            if (atoms[i].mode == 1) {
                has_blanks = 1;
                break;
            }
        }
        printf("  Pattern has %zu atoms, blanks: %s\n", num_atoms, has_blanks ? "yes" : "no");
    } else {
        printf("✗ Failed to create pattern from sequences\n");
    }
    
    explanation_free(&exp);
    explanation_free(&exp2);
    graph_destroy(g);
    graph_destroy(g2);
    
    return found_pattern ? 0 : 1;
}

