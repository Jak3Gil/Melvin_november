#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Profiling helper: get current time in seconds (monotonic clock)
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    // Create graph with reasonable capacities
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    if (!g) {
        fprintf(stderr, "Failed to create graph\n");
        return 1;
    }
    
    printf("Melvin Minimal Graph Substrate Demo\n");
    printf("===================================\n\n");
    
    // Read a line of bytes from stdin
    char buffer[1024];
    printf("Enter a string (max 1023 chars): ");
    if (!fgets(buffer, sizeof(buffer), stdin)) {
        fprintf(stderr, "Failed to read input\n");
        graph_destroy(g);
        return 1;
    }
    
    // Remove newline if present
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
        buffer[len - 1] = '\0';
        len--;
    }
    
    if (len == 0) {
        printf("Empty input, exiting.\n");
        graph_destroy(g);
        return 0;
    }
    
    printf("\nAdding %zu bytes as DATA nodes...\n", len);
    
    // Add each byte as a DATA node
    uint64_t prev_data_id = UINT64_MAX;
    for (size_t i = 0; i < len; i++) {
        Node *data_node = graph_add_data_byte(g, (uint8_t)buffer[i]);
        if (!data_node) {
            fprintf(stderr, "Failed to add DATA node at position %zu\n", i);
            graph_destroy(g);
            return 1;
        }
        
        // Add sequence edge from previous DATA to current DATA
        if (prev_data_id != UINT64_MAX) {
            graph_add_edge(g, prev_data_id, data_node->id, 1.0f);
        }
        
        prev_data_id = data_node->id;
    }
    
    printf("Created %llu DATA nodes with sequence edges.\n\n",
           (unsigned long long)g->next_data_pos);
    
    // Create a pattern for "ab"
    PatternAtom atoms[2];
    atoms[0].delta = 0;
    atoms[0].mode = 0;  // CONST_BYTE
    atoms[0].value = 'a';
    
    atoms[1].delta = 1;
    atoms[1].mode = 0;  // CONST_BYTE
    atoms[1].value = 'b';
    
    Node *pattern = graph_add_pattern(g, atoms, 2, 0.5f);
    if (!pattern) {
        fprintf(stderr, "Failed to create pattern\n");
        graph_destroy(g);
        return 1;
    }
    
    printf("Created pattern for \"ab\" (id: %llu)\n\n",
           (unsigned long long)pattern->id);
    
    // Slide pattern over DATA id range
    printf("Pattern matching results:\n");
    printf("Anchor ID | Score | Match\n");
    printf("----------|-------|------\n");
    
    int found_match = 0;
    for (uint64_t anchor_id = 0; anchor_id < g->next_data_pos; anchor_id++) {
        float score = pattern_match_score(g, pattern, anchor_id);
        if (score > 0.0f) {
            printf("%9llu | %5.2f | ", (unsigned long long)anchor_id, score);
            if (score >= 1.0f) {
                printf("FULL MATCH");
                found_match = 1;
            } else {
                printf("partial");
            }
            printf("\n");
        }
    }
    
    if (!found_match) {
        printf("(no matches found)\n");
    }
    
    printf("\n");
    
    // Print graph statistics
    graph_print_stats(g);
    
    // Optional: demonstrate propagation
    printf("\nRunning propagation (3 steps)...\n");
    double t_prop_start = now_sec();
    graph_propagate(g, 3);
    double t_prop_end = now_sec();
    printf("TIMING: propagation (3 steps) took %.3f ms\n", (t_prop_end - t_prop_start) * 1000.0);
    
    printf("Sample activations after propagation:\n");
    int shown = 0;
    for (uint64_t i = 0; i < g->num_nodes && shown < 10; i++) {
        if (g->nodes[i].kind == NODE_DATA && g->nodes[i].a != 0.0f) {
            printf("  DATA[%llu]: a=%.3f\n",
                   (unsigned long long)g->nodes[i].id,
                   g->nodes[i].a);
            shown++;
        }
    }
    
    // Phase 3: Self-consistency episodes (explanation-based)
    printf("\n=== Self-Consistency Episodes (Phase 3) ===\n");
    uint64_t start_id = 0;
    uint64_t end_id = g->next_data_pos > 0 ? g->next_data_pos - 1 : 0;
    
    float match_threshold = 0.9f;  // pattern_match_score must be at least this
    float lr_q = 0.2f;             // learning rate for pattern quality
    
    printf("Initial pattern quality: %.4f\n", pattern->q);
    printf("Running self-consistency episodes over DATA range [%llu, %llu]...\n",
           (unsigned long long)start_id,
           (unsigned long long)end_id);
    
    // Debug: Show explanation for first iteration
    double t0 = now_sec();
    Explanation exp_debug;
    explanation_init(&exp_debug);
    graph_build_explanation_single_pattern(g, pattern, start_id, end_id, match_threshold, &exp_debug);
    double t1 = now_sec();
    printf("Explanation apps: %zu (built in %.3f ms)\n", exp_debug.count, (t1 - t0) * 1000.0);
    for (size_t i = 0; i < exp_debug.count && i < 10; i++) {
        printf("  pattern %llu at anchor %llu\n",
               (unsigned long long)exp_debug.apps[i].pattern_id,
               (unsigned long long)exp_debug.apps[i].anchor_id);
    }
    if (exp_debug.count > 10) {
        printf("  ... (%zu more)\n", exp_debug.count - 10);
    }
    
    // Show reconstruction
    size_t seg_len = (size_t)(end_id - start_id + 1);
    uint8_t *pred_debug = malloc(seg_len);
    double t2 = now_sec();
    size_t written = graph_reconstruct_from_explanation(g, &exp_debug, start_id, end_id, pred_debug, seg_len);
    double t3 = now_sec();
    printf("Reconstructed %zu positions (in %.3f ms): ", written, (t3 - t2) * 1000.0);
    for (size_t i = 0; i < seg_len; i++) {
        putchar(pred_debug[i] ? pred_debug[i] : '.');
    }
    putchar('\n');
    free(pred_debug);
    explanation_free(&exp_debug);
    printf("\n");
    
    double t_episodes_start = now_sec();
    for (int it = 0; it < 5; it++) {
        double t_ep = now_sec();
        float err = graph_self_consistency_episode_single_pattern(g,
                                                                  pattern,
                                                                  start_id,
                                                                  end_id,
                                                                  match_threshold,
                                                                  lr_q);
        double t_ep_end = now_sec();
        printf("Iter %d: avg_error=%.4f, pattern_q=%.4f (took %.3f ms)\n", 
               it, err, pattern->q, (t_ep_end - t_ep) * 1000.0);
    }
    double t_episodes_end = now_sec();
    printf("TIMING: 5 episodes took %.3f ms total (avg %.3f ms/episode)\n",
           (t_episodes_end - t_episodes_start) * 1000.0,
           (t_episodes_end - t_episodes_start) * 1000.0 / 5.0);
    
    // Phase 4: Multi-pattern competition
    printf("\n=== Multi-Pattern Competition (Phase 4) ===\n");
    
    // Create additional patterns
    PatternAtom atoms_bc[2];
    atoms_bc[0].delta = 0;
    atoms_bc[0].mode = 0;
    atoms_bc[0].value = 'b';
    atoms_bc[1].delta = 1;
    atoms_bc[1].mode = 0;
    atoms_bc[1].value = 'c';
    
    Node *pattern_bc = graph_add_pattern(g, atoms_bc, 2, 0.5f);
    if (!pattern_bc) {
        fprintf(stderr, "Failed to create 'bc' pattern\n");
        graph_destroy(g);
        return 1;
    }
    
    PatternAtom atoms_aba[3];
    atoms_aba[0].delta = 0;
    atoms_aba[0].mode = 0;
    atoms_aba[0].value = 'a';
    atoms_aba[1].delta = 1;
    atoms_aba[1].mode = 0;
    atoms_aba[1].value = 'b';
    atoms_aba[2].delta = 2;
    atoms_aba[2].mode = 0;
    atoms_aba[2].value = 'a';
    
    Node *pattern_aba = graph_add_pattern(g, atoms_aba, 3, 0.5f);
    if (!pattern_aba) {
        fprintf(stderr, "Failed to create 'aba' pattern\n");
        graph_destroy(g);
        return 1;
    }
    
    // Array of patterns for multi-pattern episode
    Node *patterns[3] = {pattern, pattern_bc, pattern_aba};
    size_t num_patterns = 3;
    
    printf("Patterns: 'ab' (q=%.4f), 'bc' (q=%.4f), 'aba' (q=%.4f)\n",
           pattern->q, pattern_bc->q, pattern_aba->q);
    
    // Show candidates and selected explanation
    Explanation candidates;
    explanation_init(&candidates);
    graph_collect_candidates_multi_pattern(g, patterns, num_patterns,
                                          start_id, end_id,
                                          match_threshold, &candidates);
    printf("Candidates: %zu applications\n", candidates.count);
    
    Explanation selected;
    explanation_init(&selected);
    explanation_select_greedy_consistent(g, &candidates,
                                         start_id, end_id, &selected);
    printf("Selected: %zu applications (after conflict resolution)\n", selected.count);
    
    for (size_t i = 0; i < selected.count && i < 10; i++) {
        Node *p = graph_find_node_by_id(g, selected.apps[i].pattern_id);
        const char *name = "?";
        if (p == pattern) name = "ab";
        else if (p == pattern_bc) name = "bc";
        else if (p == pattern_aba) name = "aba";
        printf("  %s at anchor %llu\n",
               name,
               (unsigned long long)selected.apps[i].anchor_id);
    }
    if (selected.count > 10) {
        printf("  ... (%zu more)\n", selected.count - 10);
    }
    
    explanation_free(&candidates);
    explanation_free(&selected);
    
    printf("\nRunning multi-pattern episodes...\n");
    double t_multi_start = now_sec();
    for (int it = 0; it < 5; it++) {
        double t_iter = now_sec();
        float err = graph_self_consistency_episode_multi_pattern(g,
                                                                 patterns,
                                                                 num_patterns,
                                                                 start_id,
                                                                 end_id,
                                                                 match_threshold,
                                                                 lr_q);
        double t_iter_end = now_sec();
        printf("Iter %d: avg_error=%.4f, qs: ab=%.4f, bc=%.4f, aba=%.4f (took %.3f ms)\n",
               it, err, pattern->q, pattern_bc->q, pattern_aba->q, (t_iter_end - t_iter) * 1000.0);
    }
    double t_multi_end = now_sec();
    printf("TIMING: 5 multi-pattern episodes took %.3f ms total (avg %.3f ms/episode)\n",
           (t_multi_end - t_multi_start) * 1000.0,
           (t_multi_end - t_multi_start) * 1000.0 / 5.0);
    
    printf("\nAfter multi-pattern episodes, pattern bindings:\n");
    graph_debug_print_pattern_bindings(g, 16);
    
    graph_destroy(g);
    return 0;
}

