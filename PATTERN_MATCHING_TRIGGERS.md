# Pattern Matching Triggers

## Two Places Where Pattern Matching is Triggered

### 1. During Pattern Creation (`pattern_law_apply`)

**Location:** `src/melvin.c:4048-4066`

**When:** Called from `melvin_feed_byte` → `pattern_law_apply`

**What happens:**
- When a sequence is fed and a pattern already exists
- Checks if the current sequence matches any existing patterns
- If match found:
  - Strengthens the pattern (activation boost)
  - Calls `extract_and_route_to_exec` to extract values and route to EXEC nodes

**Code:**
```c
/* Pattern already exists - strengthen it and check for pattern pattern matching */
for (uint64_t i = 0; i < g->node_count; i++) {
    if (g->nodes[i].pattern_data_offset > 0) {
        /* This is a pattern node - check if sequence matches it */
        uint32_t bindings[256] = {0};
        if (pattern_matches_sequence(g, (uint32_t)i, sequence, length, bindings)) {
            /* Sequence matches existing pattern - strengthen it */
            g->nodes[i].a += 0.05f;
            prop_queue_add(g, (uint32_t)i);
            
            /* Extract values from blanks and route to EXEC nodes */
            extract_and_route_to_exec(g, (uint32_t)i, bindings);
        }
    }
}
```

### 2. During Wave Propagation (`update_node_and_propagate`)

**Location:** `src/melvin.c:2252-2270`

**When:** Called during UEL propagation when a pattern node is activated

**What happens:**
- When a pattern node's activation exceeds threshold during propagation
- Checks if the current sequence buffer matches the pattern
- If match found:
  - Activates the pattern node
  - Calls `extract_and_route_to_exec` to extract values and route to EXEC nodes

**Code:**
```c
/* GENERAL: When patterns match sequences, extract values and route to EXEC nodes */
if (g->nodes[node_id].pattern_data_offset > 0 && fabsf(new_a) > g->avg_activation * 0.5f) {
    /* Pattern node activated - check if it matches current sequence and routes to EXEC */
    if (g->sequence_buffer && g->sequence_buffer_pos > 0) {
        /* Get recent sequence */
        uint32_t seq_len = (g->sequence_buffer_pos < 20) ? (uint32_t)g->sequence_buffer_pos : 20;
        uint32_t start_pos = (g->sequence_buffer_pos >= seq_len) ? 
                             (uint32_t)(g->sequence_buffer_pos - seq_len) : 0;
        uint32_t sequence[20];
        for (uint32_t i = 0; i < seq_len; i++) {
            sequence[i] = g->sequence_buffer[(start_pos + i) % g->sequence_buffer_size];
        }
        
        /* Check if pattern matches sequence */
        uint32_t bindings[256] = {0};
        if (pattern_matches_sequence(g, node_id, sequence, seq_len, bindings)) {
            /* Pattern matches - extract values and route to EXEC */
            extract_and_route_to_exec(g, node_id, bindings);
        }
    }
}
```

## The Problem

**Pattern matching requires:**
1. A sequence buffer with the current sequence
2. Pattern nodes to exist
3. The sequence to match a pattern (similarity-based)

**For queries like "1+1=?":**
- The sequence is fed byte-by-byte via `melvin_feed_byte`
- Each byte triggers `pattern_law_apply`
- Pattern matching should happen when the full sequence is in the buffer

**Potential issues:**
1. Sequence buffer might not contain the full query when matching happens
2. Pattern matching might happen too early (before full sequence is fed)
3. Similarity threshold might be too strict
4. Sequence length mismatch (pattern length vs query length)

