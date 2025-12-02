# Pattern Discovery Redesign: Event-Driven Approach

## The Problem

Current pattern discovery scans the entire sequence buffer on every Nth byte, comparing sequences explicitly. This is **O(N²)** and slow.

## The Solution: Co-Activation Detection

Use the propagation queue to detect patterns naturally through wave dynamics!

### Key Insight:

**Patterns are sequences that co-activate repeatedly.**

When nodes A→B→C activate together multiple times:
1. They appear in the propagation queue together
2. Edges between them strengthen (Hebbian learning)
3. This signature = pattern!

### Algorithm:

```c
// In uel_main propagation loop:

while (queue has items) {
    node = dequeue();
    propagate(node);
    
    // Track recent activations (sliding window)
    recent_activations[window_pos] = node;
    window_pos = (window_pos + 1) % WINDOW_SIZE;
    
    // Every N activations, check for repeated sequences
    if (activations_count % CHECK_INTERVAL == 0) {
        detect_coactivation_patterns(recent_activations, WINDOW_SIZE);
    }
}
```

### detect_coactivation_patterns:

```c
void detect_coactivation_patterns(uint32_t *activations, int window_size) {
    // Look for sequences that appear multiple times in window
    // Example: [A, B, C, X, Y, A, B, C, Z] → "ABC" appears 2x
    
    // Use hash table for O(N) detection:
    for (int len = 3; len <= 5; len++) {
        for (int i = 0; i < window_size - len; i++) {
            uint64_t hash = hash_sequence(&activations[i], len);
            
            if (seen_before(hash)) {
                // This sequence activated twice!
                create_pattern_from_sequence(&activations[i], len);
            }
            
            mark_seen(hash);
        }
    }
}
```

## Advantages:

1. **O(window_size) not O(buffer_size²)** - Much faster!
2. **Event-driven** - Only runs when nodes activate
3. **Natural** - Patterns emerge from physics, not explicit search
4. **Scalable** - Window size is fixed (e.g., 100 recent activations)

## Implementation:

### Remove:
- ✗ `sequence_buffer` scanning
- ✗ Explicit pattern matching in `discover_patterns`
- ✗ Buffer comparison loops

### Add:
- ✓ `recent_activations` circular buffer (fixed size: 100)
- ✓ `coactivation_hash_table` for fast lookup
- ✓ Check in `uel_main` every 100 activations

### Performance:

**Before**: 16 inputs/sec (pattern scanning bottleneck)  
**After**: 1000+ inputs/sec (O(window) co-activation) ← **TARGET**

## Code Changes:

### 1. Add to Graph struct:
```c
uint32_t *recent_activations;  // Circular buffer
int activation_window_pos;
int activation_window_size;
HashTable *coactivation_patterns;  // Track seen sequences
```

### 2. In uel_main (propagation loop):
```c
// After dequeue and propagate:
g->recent_activations[g->activation_window_pos] = node_id;
g->activation_window_pos = (g->activation_window_pos + 1) % g->activation_window_size;

// Every N activations:
if (++g->activation_check_counter >= 100) {
    detect_coactivation_patterns(g);
    g->activation_check_counter = 0;
}
```

### 3. Replace pattern_law_apply:
```c
// OLD (slow):
pattern_law_apply(g, data_id);  // Scans buffer

// NEW (fast):
// Do nothing! Patterns detected in propagation loop
```

## Why This is True to UEL Physics:

- **Patterns are attractors in the energy landscape**
- **Co-activation = nodes falling into same attractor repeatedly**
- **Detection should be passive observation of physics, not active search**

This makes pattern discovery an emergent property of wave propagation, not a separate algorithm!

## Expected Speedup:

- Current: ~10-30 minutes for 10K inputs
- With this: **~10 seconds** for 10K inputs (100x faster!)

This gets us to the **thousands of inputs per second** we should have!

