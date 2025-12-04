# DYNAMIC PATTERN SIZE: Multiple Approaches

**Goal**: Let the system discover optimal pattern sizes instead of hardcoding them

---

## 🎯 APPROACH 1: Range-Based Discovery (SIMPLEST)

### Try Multiple Lengths in One Pass

**Implementation**:
```c
/* In detect_coactivation_patterns() around line 4665 */

/* Try multiple pattern lengths dynamically */
for (int len = 2; len <= 10; len++) {  // Min 2, max 10
    
    /* Only check if window has enough space */
    if (len > window_size) continue;
    
    for (int i = 0; i < window_size - len; i++) {
        uint32_t seq[10];
        
        /* Extract sequence of this length */
        for (int j = 0; j < len; j++) {
            uint32_t pos = (g->activation_window_pos + i + j) % g->activation_window_size;
            seq[j] = g->recent_activations[pos];
        }
        
        /* Rest of pattern creation... */
    }
}
```

**Pros**:
- ✅ Simple - just loop over lengths
- ✅ Discovers patterns at all sizes
- ✅ No hardcoded assumptions

**Cons**:
- ⚠️ More computation (checks all lengths)
- ⚠️ Might create redundant patterns

---

## 🎯 APPROACH 2: Adaptive Based on Data (SMARTER)

### Learn Optimal Sizes from Statistics

**Implementation**:
```c
/* Add to Graph struct */
typedef struct {
    uint32_t pattern_length;
    uint32_t usage_count;      // How often matched
    float success_rate;         // How useful
} PatternStats;

PatternStats length_stats[10];  // Track stats for each length

/* In pattern discovery */
void discover_patterns_adaptive(Graph *g) {
    /* Start with common lengths */
    int min_len = 2;
    int max_len = 10;
    
    /* Prioritize lengths that have been successful */
    int lengths[10];
    int length_count = 0;
    
    for (int len = min_len; len <= max_len; len++) {
        /* Include this length if:
         * 1. We haven't tried it much yet (exploration)
         * 2. It has good success rate (exploitation)
         */
        if (g->length_stats[len].usage_count < 100 ||  // Explore
            g->length_stats[len].success_rate > 0.5) {  // Exploit
            lengths[length_count++] = len;
        }
    }
    
    /* Try prioritized lengths */
    for (int i = 0; i < length_count; i++) {
        int len = lengths[i];
        discover_patterns_of_length(g, len);
    }
}

/* Update stats when pattern is used */
void update_pattern_stats(Graph *g, uint32_t pattern_id, bool success) {
    PatternData *pd = get_pattern_data(g, pattern_id);
    uint32_t len = pd->element_count;
    
    if (len < 10) {
        g->length_stats[len].usage_count++;
        if (success) {
            g->length_stats[len].success_rate = 
                (g->length_stats[len].success_rate * 0.95f) + (1.0f * 0.05f);
        } else {
            g->length_stats[len].success_rate *= 0.95f;
        }
    }
}
```

**Pros**:
- ✅ Learns optimal sizes from experience
- ✅ Adapts to the data
- ✅ Efficient (focuses on useful lengths)

**Cons**:
- ⚠️ More complex
- ⚠️ Needs tuning of exploration/exploitation

---

## 🎯 APPROACH 3: Frequency-Based (DATA-DRIVEN)

### Discover Based on Sequence Frequency

**Implementation**:
```c
/* Track how often sequences of each length repeat */
typedef struct {
    uint64_t hash;
    uint32_t length;
    uint32_t frequency;
} SequenceFrequency;

SequenceFrequency freq_table[1024];

/* In co-activation detection */
void detect_patterns_frequency_based(Graph *g) {
    /* For each length, track sequence frequencies */
    for (int len = 2; len <= 10; len++) {
        for (int i = 0; i < window_size - len; i++) {
            uint32_t seq[10];
            extract_sequence(g, i, len, seq);
            
            uint64_t hash = hash_sequence(seq, len);
            
            /* Look up in frequency table */
            int idx = find_or_create_entry(freq_table, hash, len);
            freq_table[idx].frequency++;
            
            /* Create pattern if frequency exceeds threshold */
            if (freq_table[idx].frequency >= get_threshold_for_length(len)) {
                create_pattern(g, seq, len);
            }
        }
    }
}

/* Threshold scales with length - longer patterns need fewer repetitions */
uint32_t get_threshold_for_length(uint32_t len) {
    if (len <= 3) return 5;   // Short patterns: need 5 repetitions
    if (len <= 5) return 3;   // Medium patterns: need 3 repetitions
    return 2;                  // Long patterns: need 2 repetitions
}
```

**Pros**:
- ✅ Data-driven (creates patterns that actually repeat)
- ✅ Longer patterns need less evidence
- ✅ Natural frequency filtering

**Cons**:
- ⚠️ More memory for frequency tracking
- ⚠️ Might miss rare but important patterns

---

## 🎯 APPROACH 4: Hierarchical (COMPOSITIONAL)

### Build Longer Patterns from Shorter Ones

**Implementation**:
```c
/* First pass: Discover short patterns (length 2-3) */
void discover_base_patterns(Graph *g) {
    for (int len = 2; len <= 3; len++) {
        // Standard discovery
    }
}

/* Second pass: Compose longer patterns from base patterns */
void compose_longer_patterns(Graph *g) {
    /* Find adjacent patterns */
    for (uint32_t p1 = 840; p1 < g->node_count; p1++) {
        if (!is_pattern(g, p1)) continue;
        
        for (uint32_t p2 = 840; p2 < g->node_count; p2++) {
            if (!is_pattern(g, p2)) continue;
            
            /* If p1 and p2 often activate sequentially, combine them */
            if (patterns_coactivate_sequentially(g, p1, p2)) {
                uint32_t composed = compose_patterns(g, p1, p2);
                // Now we have longer pattern!
            }
        }
    }
}
```

**Pros**:
- ✅ Hierarchical (matches brain structure)
- ✅ Reuses existing patterns
- ✅ Natural composition

**Cons**:
- ⚠️ Complex implementation
- ⚠️ Requires tracking sequential activation

---

## 🎯 APPROACH 5: Window-Based (ADAPTIVE TO INPUT)

### Size Based on Input Characteristics

**Implementation**:
```c
/* Analyze input to determine pattern sizes */
uint32_t estimate_pattern_size(Graph *g) {
    /* Look at recent input characteristics */
    
    /* Check spacing between operators/delimiters */
    uint32_t avg_token_length = 0;
    uint32_t token_count = 0;
    uint32_t current_token_len = 0;
    
    for (int i = 0; i < 100 && i < g->sequence_buffer_pos; i++) {
        uint32_t node_id = g->sequence_buffer[i];
        uint8_t byte = g->nodes[node_id].byte;
        
        /* Is this a delimiter/operator? */
        if (byte == '+' || byte == '-' || byte == '=' || byte == ' ') {
            if (current_token_len > 0) {
                avg_token_length += current_token_len;
                token_count++;
                current_token_len = 0;
            }
        } else {
            current_token_len++;
        }
    }
    
    if (token_count > 0) {
        avg_token_length /= token_count;
        /* Pattern size should be ~3x token length */
        return avg_token_length * 3;
    }
    
    return 5;  // Default
}

/* Use estimated size */
void discover_patterns_adaptive_size(Graph *g) {
    uint32_t estimated_size = estimate_pattern_size(g);
    
    /* Try estimated size ± 2 */
    for (int len = estimated_size - 2; len <= estimated_size + 2; len++) {
        if (len >= 2 && len <= 10) {
            discover_patterns_of_length(g, len);
        }
    }
}
```

**Pros**:
- ✅ Adapts to input structure
- ✅ Efficient (focuses on relevant sizes)
- ✅ No hardcoding

**Cons**:
- ⚠️ Needs input analysis
- ⚠️ Might miss patterns outside estimated range

---

## 💡 RECOMMENDED HYBRID APPROACH

### Combine Multiple Strategies

```c
/* Dynamic pattern discovery with multiple heuristics */
void discover_patterns_dynamic(Graph *g) {
    /* 1. Always try core sizes (2-5) - exploration */
    int core_lengths[] = {2, 3, 4, 5};
    
    /* 2. Add successful sizes from statistics - exploitation */
    for (int len = 6; len <= 10; len++) {
        if (g->length_stats[len].success_rate > 0.3f) {
            core_lengths[...] = len;
        }
    }
    
    /* 3. Estimate from recent input */
    uint32_t estimated = estimate_pattern_size(g);
    if (estimated > 5 && estimated <= 10) {
        // Add estimated size
    }
    
    /* 4. Discover at selected lengths */
    for (int i = 0; i < length_count; i++) {
        discover_patterns_of_length(g, core_lengths[i]);
    }
    
    /* 5. Limit total patterns created per discovery cycle */
    const int MAX_PATTERNS_PER_CYCLE = 10;
    // Stop after creating enough patterns
}
```

---

## 🚀 QUICK IMPLEMENTATION (5 Minutes)

### Start Simple, Then Enhance

**Step 1**: Range-based (immediate)
```c
/* Try lengths 2-7 for now */
for (int len = 2; len <= 7; len++) {
    // Discover patterns of this length
}
```

**Step 2**: Add frequency threshold (later)
```c
/* Longer patterns need less evidence */
uint32_t threshold = (len <= 3) ? 3 : 2;
if (frequency >= threshold) {
    create_pattern(...);
}
```

**Step 3**: Add statistics (future)
```c
/* Track which sizes are useful */
if (pattern_matched_successfully) {
    g->length_stats[len].success_rate += 0.1f;
}
```

---

## 📊 COMPARISON

| Approach | Complexity | Efficiency | Adaptiveness |
|----------|-----------|------------|--------------|
| **Range-Based** | Low | Medium | None |
| **Adaptive Stats** | Medium | High | Good |
| **Frequency-Based** | Medium | High | Good |
| **Hierarchical** | High | High | Excellent |
| **Window-Based** | Medium | Medium | Good |
| **Hybrid** | Medium | High | Excellent |

---

## 🎯 IMMEDIATE ACTION

**Let's implement Range-Based now** (simplest, works immediately):

```c
/* Line ~4665 in detect_coactivation_patterns() */

// Remove fixed length:
// int len = 3;

// Add dynamic range:
for (int len = 2; len <= 7; len++) {
    // Limit patterns per length
    int patterns_at_this_length = 0;
    const int MAX_PER_LENGTH = 3;
    
    for (int i = 0; i < window_size - len; i++) {
        // ... rest of discovery code ...
        
        if (pattern_created) {
            patterns_at_this_length++;
            if (patterns_at_this_length >= MAX_PER_LENGTH) {
                break;  // Move to next length
            }
        }
    }
}
```

This gives us:
- ✅ Dynamic sizing (2-7)
- ✅ All useful sizes covered
- ✅ Limited explosion (max 3 patterns per length)
- ✅ Simple to implement

**Want me to implement this right now?** 🚀


