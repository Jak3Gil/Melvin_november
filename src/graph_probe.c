#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    Graph *g;
    Node  *patterns[8];
    size_t num_patterns;
} ProbeEnv;

ProbeEnv *probe_env_create(void) {
    ProbeEnv *env = calloc(1, sizeof(ProbeEnv));
    if (!env) return NULL;
    
    env->g = graph_create(1024, 2048, 16 * 1024);
    if (!env->g) {
        free(env);
        return NULL;
    }
    
    env->num_patterns = 0;
    for (size_t i = 0; i < 8; i++) {
        env->patterns[i] = NULL;
    }
    
    return env;
}

void probe_env_destroy(ProbeEnv *env) {
    if (!env) return;
    if (env->g) {
        graph_destroy(env->g);
    }
    free(env);
}

void probe_feed_string(ProbeEnv *env, const char *s) {
    if (!env || !env->g || !s) return;
    
    // Add each byte as a DATA node
    uint64_t prev_data_id = UINT64_MAX;
    for (const char *p = s; *p; p++) {
        Node *data_node = graph_add_data_byte(env->g, (uint8_t)*p);
        if (data_node) {
            // Add sequence edge from previous DATA to current DATA
            if (prev_data_id != UINT64_MAX) {
                graph_add_edge(env->g, prev_data_id, data_node->id, 1.0f);
            }
            prev_data_id = data_node->id;
        }
    }
}

Node *probe_create_pattern_from_literal(ProbeEnv *env,
                                         const char *literal,
                                         int start_delta) {
    if (!env || !env->g || !literal || env->num_patterns >= 8) return NULL;
    
    size_t len = strlen(literal);
    if (len == 0) return NULL;
    
    PatternAtom *atoms = malloc(len * sizeof(PatternAtom));
    if (!atoms) return NULL;
    
    for (size_t i = 0; i < len; i++) {
        atoms[i].delta = start_delta + (int16_t)i;
        atoms[i].mode = 0;  // CONST_BYTE
        atoms[i].value = (uint8_t)literal[i];
    }
    
    Node *pattern = graph_add_pattern(env->g, atoms, len, 0.5f);
    free(atoms);
    
    if (pattern) {
        env->patterns[env->num_patterns++] = pattern;
    }
    
    return pattern;
}

void probe_experiment_functional_patterns(void) {
    printf("=== Experiment 1: Functional Patterns ===\n\n");
    
    // Test with "cababc"
    printf("Test A: Input 'cababc'\n");
    ProbeEnv *env1 = probe_env_create();
    if (!env1) {
        printf("Failed to create probe environment\n");
        return;
    }
    
    probe_feed_string(env1, "cababc");
    Node *P_ab = probe_create_pattern_from_literal(env1, "ab", 0);
    Node *P_az = probe_create_pattern_from_literal(env1, "az", 0);
    
    if (!P_ab || !P_az) {
        printf("Failed to create patterns\n");
        probe_env_destroy(env1);
        return;
    }
    
    Node *patterns[] = {P_ab, P_az};
    size_t num_patterns = 2;
    uint64_t start_id = 0;
    uint64_t end_id = env1->g->next_data_pos > 0 ? env1->g->next_data_pos - 1 : 0;
    
    printf("Running 10 multi-pattern episodes...\n");
    for (int it = 0; it < 10; it++) {
        graph_self_consistency_episode_multi_pattern(env1->g,
                                                     patterns,
                                                     num_patterns,
                                                     start_id,
                                                     end_id,
                                                     0.8f,   // match_threshold
                                                     0.2f);  // lr_q
    }
    
    float q_ab_good = P_ab->q;
    float q_az_good = P_az->q;
    
    printf("After training on 'cababc':\n");
    printf("  P_ab->q = %.4f\n", q_ab_good);
    printf("  P_az->q = %.4f\n", q_az_good);
    printf("\nBindings:\n");
    graph_debug_print_pattern_bindings(env1->g, 16);
    
    probe_env_destroy(env1);
    
    // Test with "xxxxxx"
    printf("\nTest B: Input 'xxxxxx'\n");
    ProbeEnv *env2 = probe_env_create();
    if (!env2) {
        printf("Failed to create probe environment\n");
        return;
    }
    
    probe_feed_string(env2, "xxxxxx");
    Node *P_ab2 = probe_create_pattern_from_literal(env2, "ab", 0);
    Node *P_az2 = probe_create_pattern_from_literal(env2, "az", 0);
    
    if (!P_ab2 || !P_az2) {
        printf("Failed to create patterns\n");
        probe_env_destroy(env2);
        return;
    }
    
    Node *patterns2[] = {P_ab2, P_az2};
    num_patterns = 2;
    start_id = 0;
    end_id = env2->g->next_data_pos > 0 ? env2->g->next_data_pos - 1 : 0;
    
    printf("Running 10 multi-pattern episodes...\n");
    for (int it = 0; it < 10; it++) {
        graph_self_consistency_episode_multi_pattern(env2->g,
                                                      patterns2,
                                                      num_patterns,
                                                      start_id,
                                                      end_id,
                                                      0.8f,
                                                      0.2f);
    }
    
    float q_ab_noise = P_ab2->q;
    float q_az_noise = P_az2->q;
    
    printf("After training on 'xxxxxx':\n");
    printf("  P_ab->q = %.4f\n", q_ab_noise);
    printf("  P_az->q = %.4f\n", q_az_noise);
    printf("\nBindings:\n");
    graph_debug_print_pattern_bindings(env2->g, 16);
    
    probe_env_destroy(env2);
    
    // Metric
    printf("\nFunctional Pattern Score:\n");
    printf("  P_ab: q_after_good - q_after_noise = %.4f - %.4f = %.4f\n",
           q_ab_good, q_ab_noise, q_ab_good - q_ab_noise);
    printf("  P_az: q_after_good - q_after_noise = %.4f - %.4f = %.4f\n",
           q_az_good, q_az_noise, q_az_good - q_az_noise);
    
    if (q_ab_good > q_ab_noise && q_ab_good > 0.6f) {
        printf("  ✓ P_ab shows functional pattern behavior\n");
    } else {
        printf("  ✗ P_ab does not show clear functional pattern behavior\n");
    }
}

void probe_experiment_compression_and_accuracy(void) {
    printf("=== Experiment 2: Compression & Accuracy ===\n\n");
    
    const char *test_strings[] = {"ababab", "abcabcabc", "randomnoise"};
    size_t num_tests = 3;
    
    for (size_t t = 0; t < num_tests; t++) {
        const char *s = test_strings[t];
        printf("Input: \"%s\"\n", s);
        
        ProbeEnv *env = probe_env_create();
        if (!env) {
            printf("Failed to create probe environment\n");
            continue;
        }
        
        probe_feed_string(env, s);
        
        // Create patterns based on string
        Node *patterns[4];
        size_t num_patterns = 0;
        
        if (strcmp(s, "ababab") == 0) {
            patterns[num_patterns++] = probe_create_pattern_from_literal(env, "ab", 0);
            patterns[num_patterns++] = probe_create_pattern_from_literal(env, "aba", 0);
        } else if (strcmp(s, "abcabcabc") == 0) {
            patterns[num_patterns++] = probe_create_pattern_from_literal(env, "abc", 0);
            patterns[num_patterns++] = probe_create_pattern_from_literal(env, "ab", 0);
        } else {
            // For random noise, try some generic patterns
            patterns[num_patterns++] = probe_create_pattern_from_literal(env, "ab", 0);
            patterns[num_patterns++] = probe_create_pattern_from_literal(env, "cd", 0);
        }
        
        if (num_patterns == 0) {
            printf("Failed to create patterns\n");
            probe_env_destroy(env);
            continue;
        }
        
        // Run training episodes
        uint64_t start_id = 0;
        uint64_t end_id = env->g->next_data_pos > 0 ? env->g->next_data_pos - 1 : 0;
        
        for (int it = 0; it < 10; it++) {
            graph_self_consistency_episode_multi_pattern(env->g,
                                                         patterns,
                                                         num_patterns,
                                                         start_id,
                                                         end_id,
                                                         0.8f,
                                                         0.2f);
        }
        
        // Build explanation without further learning
        Explanation candidates, selected;
        explanation_init(&candidates);
        explanation_init(&selected);
        
        graph_collect_candidates_multi_pattern(env->g,
                                              patterns,
                                              num_patterns,
                                              start_id,
                                              end_id,
                                              0.8f,
                                              &candidates);
        
        explanation_select_greedy_consistent(env->g,
                                             &candidates,
                                             start_id,
                                             end_id,
                                             &selected);
        
        // Reconstruct
        size_t segment_len = (size_t)(end_id - start_id + 1);
        uint8_t *actual = malloc(segment_len);
        uint8_t *pred = malloc(segment_len);
        
        if (actual && pred) {
            graph_collect_data_span(env->g, start_id, actual, segment_len);
            graph_reconstruct_from_explanation(env->g, &selected,
                                              start_id, end_id,
                                              pred, segment_len);
            
            // Compute metrics
            size_t positions = 0;
            size_t errors = 0;
            for (size_t i = 0; i < segment_len; i++) {
                if (pred[i] == 0x00) continue;
                positions++;
                if (pred[i] != actual[i]) {
                    errors++;
                }
            }
            
            float compression_ratio = 0.0f;
            if (segment_len > 0) {
                compression_ratio = (float)selected.count / (float)segment_len;
            }
            
            float reconstruction_error = 0.0f;
            if (positions > 0) {
                reconstruction_error = (float)errors / (float)positions;
            } else if (selected.count == 0) {
                // No predictions made, can't assess
                reconstruction_error = 1.0f;
            }
            
            printf("  Segment length: %zu\n", segment_len);
            printf("  Explanation apps: %zu\n", selected.count);
            printf("  Compression ratio: %.3f\n", compression_ratio);
            printf("  Reconstruction error: %.3f\n", reconstruction_error);
            
            // Intelligent if: compression achieved AND low error AND actually made predictions
            int intelligent = (selected.count > 0 && 
                              compression_ratio < 1.0f && 
                              reconstruction_error < 0.1f) ? 1 : 0;
            printf("  Intelligent: %s\n", intelligent ? "YES" : "NO");
        }
        
        free(actual);
        free(pred);
        explanation_free(&candidates);
        explanation_free(&selected);
        probe_env_destroy(env);
        printf("\n");
    }
}

void probe_experiment_generalization(void) {
    printf("=== Experiment 3: Generalization ===\n\n");
    
    // Training phase
    printf("Training phase:\n");
    ProbeEnv *train_env = probe_env_create();
    if (!train_env) {
        printf("Failed to create training environment\n");
        return;
    }
    
    probe_feed_string(train_env, "ababab");
    Node *P_ab_train = probe_create_pattern_from_literal(train_env, "ab", 0);
    Node *P_aba_train = probe_create_pattern_from_literal(train_env, "aba", 0);
    
    if (!P_ab_train || !P_aba_train) {
        printf("Failed to create training patterns\n");
        probe_env_destroy(train_env);
        return;
    }
    
    Node *train_patterns[] = {P_ab_train, P_aba_train};
    size_t num_train_patterns = 2;
    uint64_t start_id = 0;
    uint64_t end_id = train_env->g->next_data_pos > 0 ? train_env->g->next_data_pos - 1 : 0;
    
    printf("Training on 'ababab'...\n");
    for (int it = 0; it < 10; it++) {
        graph_self_consistency_episode_multi_pattern(train_env->g,
                                                      train_patterns,
                                                      num_train_patterns,
                                                      start_id,
                                                      end_id,
                                                      0.8f,
                                                      0.2f);
    }
    
    float q_ab_trained = P_ab_train->q;
    float q_aba_trained = P_aba_train->q;
    
    printf("After training:\n");
    printf("  P_ab->q = %.4f\n", q_ab_trained);
    printf("  P_aba->q = %.4f\n", q_aba_trained);
    
    // Also train on "cababc"
    probe_feed_string(train_env, "cababc");
    end_id = train_env->g->next_data_pos > 0 ? train_env->g->next_data_pos - 1 : 0;
    
    printf("Training on 'cababc'...\n");
    for (int it = 0; it < 10; it++) {
        graph_self_consistency_episode_multi_pattern(train_env->g,
                                                      train_patterns,
                                                      num_train_patterns,
                                                      start_id,
                                                      end_id,
                                                      0.8f,
                                                      0.2f);
    }
    
    q_ab_trained = P_ab_train->q;
    q_aba_trained = P_aba_train->q;
    
    printf("After additional training:\n");
    printf("  P_ab->q = %.4f\n", q_ab_trained);
    printf("  P_aba->q = %.4f\n", q_aba_trained);
    
    probe_env_destroy(train_env);
    
    // Test phase
    printf("\nTest phase:\n");
    ProbeEnv *test_env = probe_env_create();
    if (!test_env) {
        printf("Failed to create test environment\n");
        return;
    }
    
    probe_feed_string(test_env, "ababababa");
    printf("Test on 'ababababa'\n");
    
    // Recreate patterns with trained q values
    Node *P_ab_test = probe_create_pattern_from_literal(test_env, "ab", 0);
    Node *P_aba_test = probe_create_pattern_from_literal(test_env, "aba", 0);
    
    if (!P_ab_test || !P_aba_test) {
        printf("Failed to create test patterns\n");
        probe_env_destroy(test_env);
        return;
    }
    
    // Set initial q to trained values
    P_ab_test->q = q_ab_trained;
    P_aba_test->q = q_aba_trained;
    
    Node *test_patterns[] = {P_ab_test, P_aba_test};
    size_t num_test_patterns = 2;
    start_id = 0;
    end_id = test_env->g->next_data_pos > 0 ? test_env->g->next_data_pos - 1 : 0;
    
    // Build explanation without learning
    Explanation candidates, selected;
    explanation_init(&candidates);
    explanation_init(&selected);
    
    graph_collect_candidates_multi_pattern(test_env->g,
                                          test_patterns,
                                          num_test_patterns,
                                          start_id,
                                          end_id,
                                          0.8f,
                                          &candidates);
    
    explanation_select_greedy_consistent(test_env->g,
                                         &candidates,
                                         start_id,
                                         end_id,
                                         &selected);
    
    // Reconstruct
    size_t segment_len = (size_t)(end_id - start_id + 1);
    uint8_t *actual = malloc(segment_len);
    uint8_t *pred = malloc(segment_len);
    
    if (actual && pred) {
        graph_collect_data_span(test_env->g, start_id, actual, segment_len);
        graph_reconstruct_from_explanation(test_env->g, &selected,
                                          start_id, end_id,
                                          pred, segment_len);
        
        // Compute metrics
        size_t positions = 0;
        size_t errors = 0;
        for (size_t i = 0; i < segment_len; i++) {
            if (pred[i] == 0x00) continue;
            positions++;
            if (pred[i] != actual[i]) {
                errors++;
            }
        }
        
        float compression_ratio = 0.0f;
        if (segment_len > 0) {
            compression_ratio = (float)selected.count / (float)segment_len;
        }
        
        float reconstruction_error = 0.0f;
        if (positions > 0) {
            reconstruction_error = (float)errors / (float)positions;
        }
        
        printf("Generalization test:\n");
        printf("  Train on: \"ababab\", \"cababc\"\n");
        printf("  Test on: \"ababababa\"\n");
        printf("  Compression ratio (test): %.3f\n", compression_ratio);
        printf("  Reconstruction error (test): %.3f\n", reconstruction_error);
        
        if (compression_ratio < 1.0f && reconstruction_error < 0.1f) {
            printf("  ✓ Patterns generalize successfully\n");
        } else {
            printf("  ✗ Patterns do not generalize well\n");
        }
    }
    
    free(actual);
    free(pred);
    explanation_free(&candidates);
    explanation_free(&selected);
    probe_env_destroy(test_env);
}

int main(void) {
    printf("=== Melvin Graph State Probe ===\n\n");
    
    probe_experiment_functional_patterns();
    printf("\n--------------------------------\n\n");
    
    probe_experiment_compression_and_accuracy();
    printf("\n--------------------------------\n\n");
    
    probe_experiment_generalization();
    
    printf("\n=== Probe complete ===\n");
    return 0;
}

