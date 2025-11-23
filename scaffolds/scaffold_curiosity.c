// Curiosity Boundary Scaffold
// This file encodes intrinsic curiosity / exploration: reward novelty,
// penalize repeated trivial loops with no new information.
// After parsing, this file will be deleted and patterns stored in melvin.m

// Reward encountering novel states or configurations
// PATTERN_RULE(name="NOVEL_STATE_REWARD",
//   context={ state:ANY, novelty:HIGH },
//   effect={ reward:+2, encourage_explore:STATE })

// Small reward for mild surprise (learning opportunity)
// PATTERN_RULE(name="MILD_SURPRISE_REWARD",
//   context={ state:ANY, surprise:MEDIUM },
//   effect={ reward:+1 })

// Penalize degenerate looping with no information gain
// PATTERN_RULE(name="DEGENERATE_LOOP_PENALTY",
//   context={ state:ANY, looping:TRUE, information_gain:LOW },
//   effect={ reward:-2, discourage_loop:STATE })

// Reward high information gain states
// PATTERN_RULE(name="INFORMATION_GAIN_REWARD",
//   context={ state:ANY, information_gain:HIGH, surprise:MEDIUM },
//   effect={ reward:+2, encourage_explore:STATE })

// Penalize pure chaos (very high surprise with low prediction confidence)
// PATTERN_RULE(name="PURE_CHAOS_PENALTY",
//   context={ state:ANY, surprise:VERY_HIGH, prediction_confidence:LOW },
//   effect={ reward:-1 })

// Reward exploration of unexplored state space
// PATTERN_RULE(name="UNEXPLORED_STATE_REWARD",
//   context={ state:ANY, novelty:MEDIUM, exploration_value:HIGH },
//   effect={ reward:+1, encourage_explore:STATE })

// Penalize repetitive trivial patterns with no learning
// PATTERN_RULE(name="TRIVIAL_REPETITION_PENALTY",
//   context={ state:ANY, looping:TRUE, information_gain:ZERO, repetition_count:HIGH },
//   effect={ reward:-1, discourage_loop:STATE })

// Reward states that lead to pattern formation or refinement
// PATTERN_RULE(name="PATTERN_FORMATION_REWARD",
//   context={ state:ANY, novelty:MEDIUM, pattern_formation:TRUE },
//   effect={ reward:+1 })

// Empty function body - scaffold is just for pattern injection
void scaffold_curiosity(void) {
    // This function body is ignored - only comments are parsed
}

