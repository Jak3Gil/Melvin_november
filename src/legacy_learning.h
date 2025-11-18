#ifndef LEGACY_LEARNING_H
#define LEGACY_LEARNING_H

#include "melvin.h"

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

// Legacy multi-pattern candidate collection
// WARNING: Performs O(patterns × anchors) scan - only use in training mode
void legacy_collect_candidates_multi_pattern(const Graph *g,
                                            Node *const *patterns,
                                            size_t num_patterns,
                                            uint64_t start_id,
                                            uint64_t end_id,
                                            float match_threshold,
                                            Explanation *out_candidates);

// Legacy multi-pattern self-consistency episode
// WARNING: Performs O(patterns × anchors) scan - only use in training mode
float legacy_self_consistency_episode_multi_pattern(Graph *g,
                                                   Node *const *patterns,
                                                   size_t num_patterns,
                                                   uint64_t start_id,
                                                   uint64_t end_id,
                                                   float match_threshold,
                                                   float lr_q);

#endif // LEGACY_LEARNING_H

