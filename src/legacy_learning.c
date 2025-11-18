/*
 * LEGACY GLOBAL LEARNING (TO BE REPLACED)
 *
 * This file contains non-local, O(patterns × anchors) learning code.
 * It MUST NOT run in runtime mode.
 * Training-enabled runs may call this, but it is a known scalability risk.
 *
 * These functions perform global scans over patterns and anchors,
 * violating the "C is frozen hardware, graph is the brain" principle.
 * They are kept here for backward compatibility during training,
 * but should eventually be replaced with graph-native, local learning rules.
 */

#include "legacy_learning.h"
#include <stdlib.h>
#include <string.h>

extern SystemConfig g_sys;

// Helper: compute application score for sorting
static float compute_app_score(const Graph *g,
                                uint64_t pattern_id,
                                uint64_t anchor_id,
                                float epsilon) {
    Node *pattern = graph_find_node_by_id((Graph *)g, pattern_id);
    if (!pattern || pattern->kind != NODE_PATTERN) return 0.0f;
    
    float match_score = pattern_match_score(g, pattern, anchor_id);
    float q = pattern->q;
    
    return match_score * (epsilon + q);
}

void legacy_collect_candidates_multi_pattern(const Graph *g,
                                            Node *const *patterns,
                                            size_t num_patterns,
                                            uint64_t start_id,
                                            uint64_t end_id,
                                            float match_threshold,
                                            Explanation *out_candidates) {
    if (!g || !patterns || !out_candidates) return;
    if (start_id > end_id) return;
    
    // Runtime mode: skip heavy learning work
    if (!g_sys.training_enabled) {
        return;  // Fast inference mode - no pattern scanning
    }
    
    for (size_t p = 0; p < num_patterns; p++) {
        Node *pattern = patterns[p];
        if (!pattern || pattern->kind != NODE_PATTERN) continue;
        
        for (uint64_t anchor_id = start_id; anchor_id <= end_id; anchor_id++) {
            float score = pattern_match_score(g, pattern, anchor_id);
            if (score >= match_threshold) {
                explanation_add(out_candidates, pattern->id, anchor_id);
            }
        }
    }
}

float legacy_self_consistency_episode_multi_pattern(Graph *g,
                                                   Node *const *patterns,
                                                   size_t num_patterns,
                                                   uint64_t start_id,
                                                   uint64_t end_id,
                                                   float match_threshold,
                                                   float lr_q) {
    if (!g || !patterns || num_patterns == 0) return 0.0f;
    if (start_id > end_id) return 0.0f;
    
    // Runtime mode: skip heavy learning work
    if (!g_sys.training_enabled) {
        return 0.0f;  // Fast inference mode - no learning episodes
    }
    
    size_t len = (size_t)(end_id - start_id + 1);
    if (len == 0) return 0.0f;
    
    // 1) Collect actual bytes
    uint8_t *actual = malloc(len);
    if (!actual) return 0.0f;
    
    size_t got = graph_collect_data_span(g, start_id, actual, len);
    if (got < len) len = got;
    if (len == 0) {
        free(actual);
        return 0.0f;
    }
    
    // 2) Collect candidates from all patterns
    Explanation candidates;
    explanation_init(&candidates);
    legacy_collect_candidates_multi_pattern(g, patterns, num_patterns,
                                          start_id, end_id,
                                          match_threshold, &candidates);
    
    // 3) Select consistent subset
    Explanation selected;
    explanation_init(&selected);
    explanation_select_greedy_consistent(g, &candidates,
                                        start_id, end_id, &selected);
    
    // 4) Reconstruct from selected explanation
    uint8_t *pred = malloc(len);
    if (!pred) {
        explanation_free(&candidates);
        explanation_free(&selected);
        free(actual);
        return 0.0f;
    }
    
    graph_reconstruct_from_explanation(g, &selected,
                                      start_id, end_id,
                                      pred, len);
    
    // 5) Compute global error
    size_t positions = 0;
    size_t errors = 0;
    for (size_t i = 0; i < len; i++) {
        if (pred[i] != 0x00) {
            positions++;
            if (pred[i] != actual[i]) {
                errors++;
            }
        }
    }
    
    float avg_error = (positions > 0) ? (float)errors / (float)positions : 1.0f;
    
    // 6) Update each pattern's quality based on its contribution
    // For simplicity, update all patterns equally based on global error
    // (A more sophisticated version would track per-pattern contributions)
    for (size_t p = 0; p < num_patterns; p++) {
        Node *pattern = patterns[p];
        if (!pattern || pattern->kind != NODE_PATTERN) continue;
        
        // Update quality toward (1 - error)
        float target_q = 1.0f - avg_error;
        pattern->q += lr_q * (target_q - pattern->q);
        if (pattern->q < 0.0f) pattern->q = 0.0f;
        if (pattern->q > 1.0f) pattern->q = 1.0f;
    }
    
    explanation_free(&candidates);
    explanation_free(&selected);
    free(actual);
    free(pred);
    
    return avg_error; // in [0,1], 0 = perfect consistency
}

