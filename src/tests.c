#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

// Test A: Perfect explainer - "ab" pattern on "cababc"
void test_ab_on_cababc(void) {
    printf("Test A: 'ab' pattern on 'cababc'\n");
    
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    assert(g != NULL);
    
    const char *s = "cababc";
    for (const char *p = s; *p; p++) {
        Node *n = graph_add_data_byte(g, (uint8_t)*p);
        assert(n != NULL);
    }
    
    PatternAtom atoms[2];
    atoms[0].delta = 0;
    atoms[0].mode = 0;  // CONST_BYTE
    atoms[0].value = 'a';
    
    atoms[1].delta = 1;
    atoms[1].mode = 0;  // CONST_BYTE
    atoms[1].value = 'b';
    
    Node *pattern = graph_add_pattern(g, atoms, 2, 0.0f);
    assert(pattern != NULL);
    
    uint64_t start_id = 0;
    uint64_t end_id = g->next_data_pos > 0 ? g->next_data_pos - 1 : 0;
    
    float initial_q = pattern->q;
    float err = graph_self_consistency_episode_single_pattern(
        g, pattern, start_id, end_id, 0.9f, 0.2f
    );
    
    printf("  Error: %.4f, Quality: %.4f -> %.4f\n", err, initial_q, pattern->q);
    assert(err < 0.01f);  // Should be near 0
    assert(pattern->q > initial_q);  // Quality should increase
    
    graph_destroy(g);
    printf("  PASS\n\n");
}

// Test B: Irrelevant pattern - "ab" pattern on "xxxxxx"
void test_ab_on_xxxxxx(void) {
    printf("Test B: 'ab' pattern on 'xxxxxx'\n");
    
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    assert(g != NULL);
    
    const char *s = "xxxxxx";
    for (const char *p = s; *p; p++) {
        Node *n = graph_add_data_byte(g, (uint8_t)*p);
        assert(n != NULL);
    }
    
    PatternAtom atoms[2];
    atoms[0].delta = 0;
    atoms[0].mode = 0;
    atoms[0].value = 'a';
    
    atoms[1].delta = 1;
    atoms[1].mode = 0;
    atoms[1].value = 'b';
    
    Node *pattern = graph_add_pattern(g, atoms, 2, 0.5f);
    assert(pattern != NULL);
    
    uint64_t start_id = 0;
    uint64_t end_id = g->next_data_pos > 0 ? g->next_data_pos - 1 : 0;
    
    float initial_q = pattern->q;
    float err = graph_self_consistency_episode_single_pattern(
        g, pattern, start_id, end_id, 0.9f, 0.2f
    );
    
    printf("  Error: %.4f, Quality: %.4f -> %.4f\n", err, initial_q, pattern->q);
    // No matches, so quality should remain unchanged
    assert(pattern->q == initial_q);
    
    graph_destroy(g);
    printf("  PASS\n\n");
}

// Test C: Wrong pattern - "az" pattern on "cababc" (should have high error)
void test_az_on_cababc(void) {
    printf("Test C: 'az' pattern on 'cababc'\n");
    
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    assert(g != NULL);
    
    const char *s = "cababc";
    for (const char *p = s; *p; p++) {
        Node *n = graph_add_data_byte(g, (uint8_t)*p);
        assert(n != NULL);
    }
    
    PatternAtom atoms[2];
    atoms[0].delta = 0;
    atoms[0].mode = 0;
    atoms[0].value = 'a';
    
    atoms[1].delta = 1;
    atoms[1].mode = 0;
    atoms[1].value = 'z';  // wrong on purpose
    
    Node *pattern = graph_add_pattern(g, atoms, 2, 0.5f);
    assert(pattern != NULL);
    
    uint64_t start_id = 0;
    uint64_t end_id = g->next_data_pos > 0 ? g->next_data_pos - 1 : 0;
    
    float initial_q = pattern->q;
    float err = graph_self_consistency_episode_single_pattern(
        g, pattern, start_id, end_id, 0.5f, 0.2f  // Lower threshold to allow matches
    );
    
    printf("  Error: %.4f, Quality: %.4f -> %.4f\n", err, initial_q, pattern->q);
    
    // If pattern matches but is wrong, error should be high
    // If it doesn't match at all, quality stays same
    // Either way, quality should not increase significantly
    if (err > 0.01f) {
        assert(pattern->q <= initial_q || pattern->q < 0.6f);  // Should drop or stay low
    } else {
        // No matches, quality unchanged
        assert(pattern->q == initial_q);
    }
    
    graph_destroy(g);
    printf("  PASS\n\n");
}

// Test D: Multiple anchors explanation - "ab" on "ababab"
void test_explanation_multiple_anchors(void) {
    printf("Test D: Multiple anchors explanation\n");
    
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    assert(g != NULL);
    
    const char *s = "ababab";
    for (const char *p = s; *p; p++) {
        Node *n = graph_add_data_byte(g, (uint8_t)*p);
        assert(n != NULL);
    }
    
    PatternAtom atoms[2];
    atoms[0].delta = 0;
    atoms[0].mode = 0;
    atoms[0].value = 'a';
    
    atoms[1].delta = 1;
    atoms[1].mode = 0;
    atoms[1].value = 'b';
    
    Node *pattern = graph_add_pattern(g, atoms, 2, 0.5f);
    assert(pattern != NULL);
    
    uint64_t start_id = 0;
    uint64_t end_id = g->next_data_pos > 0 ? g->next_data_pos - 1 : 0;
    
    Explanation exp;
    explanation_init(&exp);
    graph_build_explanation_single_pattern(g, pattern, start_id, end_id, 0.9f, &exp);
    
    printf("  Explanation apps: %zu\n", exp.count);
    assert(exp.count == 3);  // Should have 3 matches at anchors 0, 2, 4
    
    // Check anchors
    assert(exp.apps[0].anchor_id == 0);
    assert(exp.apps[1].anchor_id == 2);
    assert(exp.apps[2].anchor_id == 4);
    
    // Reconstruct
    size_t seg_len = (size_t)(end_id - start_id + 1);
    uint8_t *pred = malloc(seg_len);
    size_t written = graph_reconstruct_from_explanation(g, &exp, start_id, end_id, pred, seg_len);
    
    printf("  Reconstructed %zu positions\n", written);
    assert(written == 6);  // Should reconstruct all 6 positions
    
    // Verify reconstruction matches original
    for (size_t i = 0; i < seg_len; i++) {
        assert(pred[i] == (uint8_t)s[i]);
    }
    
    free(pred);
    explanation_free(&exp);
    graph_destroy(g);
    printf("  PASS\n\n");
}

// Test E: Multi-pattern competition
void test_multi_pattern_competition(void) {
    printf("Test E: Multi-pattern competition\n");
    
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    assert(g != NULL);
    
    const char *s = "ababc";
    for (const char *p = s; *p; p++) {
        Node *n = graph_add_data_byte(g, (uint8_t)*p);
        assert(n != NULL);
    }
    
    // Pattern "ab"
    PatternAtom atoms_ab[2];
    atoms_ab[0].delta = 0;
    atoms_ab[0].mode = 0;
    atoms_ab[0].value = 'a';
    atoms_ab[1].delta = 1;
    atoms_ab[1].mode = 0;
    atoms_ab[1].value = 'b';
    Node *pattern_ab = graph_add_pattern(g, atoms_ab, 2, 0.5f);
    assert(pattern_ab != NULL);
    
    // Pattern "bc"
    PatternAtom atoms_bc[2];
    atoms_bc[0].delta = 0;
    atoms_bc[0].mode = 0;
    atoms_bc[0].value = 'b';
    atoms_bc[1].delta = 1;
    atoms_bc[1].mode = 0;
    atoms_bc[1].value = 'c';
    Node *pattern_bc = graph_add_pattern(g, atoms_bc, 2, 0.5f);
    assert(pattern_bc != NULL);
    
    Node *patterns[2] = {pattern_ab, pattern_bc};
    size_t num_patterns = 2;
    
    uint64_t start_id = 0;
    uint64_t end_id = g->next_data_pos > 0 ? g->next_data_pos - 1 : 0;
    
    // Test candidate collection
    Explanation candidates;
    explanation_init(&candidates);
    graph_collect_candidates_multi_pattern(g, patterns, num_patterns,
                                          start_id, end_id, 0.9f, &candidates);
    printf("  Candidates: %zu\n", candidates.count);
    assert(candidates.count >= 2);  // Should have at least "ab" at 0 and "bc" at 3
    
    // Test selection
    Explanation selected;
    explanation_init(&selected);
    explanation_select_greedy_consistent(g, &candidates,
                                         start_id, end_id, &selected);
    printf("  Selected: %zu\n", selected.count);
    assert(selected.count >= 2);  // Both should be selected (no conflicts)
    
    // Test multi-pattern episode
    float initial_q_ab = pattern_ab->q;
    float initial_q_bc = pattern_bc->q;
    
    float err = graph_self_consistency_episode_multi_pattern(g,
                                                             patterns,
                                                             num_patterns,
                                                             start_id,
                                                             end_id,
                                                             0.9f,
                                                             0.2f);
    
    printf("  Error: %.4f, q_ab: %.4f->%.4f, q_bc: %.4f->%.4f\n",
           err, initial_q_ab, pattern_ab->q, initial_q_bc, pattern_bc->q);
    
    assert(err < 0.01f);  // Should be near 0
    assert(pattern_ab->q > initial_q_ab);  // Should improve
    assert(pattern_bc->q > initial_q_bc);  // Should improve
    
    explanation_free(&candidates);
    explanation_free(&selected);
    graph_destroy(g);
    printf("  PASS\n\n");
}

int main(void) {
    printf("=== Running Melvin Phase 3 & 4 Tests ===\n\n");
    
    test_ab_on_cababc();
    test_ab_on_xxxxxx();
    test_az_on_cababc();
    test_explanation_multiple_anchors();
    test_multi_pattern_competition();
    
    printf("All tests passed!\n");
    return 0;
}

