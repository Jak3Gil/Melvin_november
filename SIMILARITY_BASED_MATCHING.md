# Similarity-Based Pattern Matching

## Philosophy

**Pattern matching uses similarity, not exact matches:**
- Node similarity: `uel_kernel` + activation similarity
- Position similarity: Same positions should have similar nodes
- Dynamic threshold: Scales with graph state (relative)
- Partial matches: Don't require 100% accuracy

## Implementation

### Dynamic Threshold

**Scales with graph state:**
```c
float similarity_threshold = avg_edge_strength * 0.3f + avg_activation * 0.2f;
```

**Adjusted by pattern strength:**
- Stronger patterns: Lower threshold (easier to match)
- Weaker patterns: Higher threshold (harder to match)
- More frequent patterns: Lower threshold (easier to match)

**Allows partial matches:**
- Minimum: 0.05 (very lenient)
- Maximum: 0.9 (allows partial matches)

### Similarity Computation

**For data nodes:**
1. Exact match: similarity = 1.0
2. Kernel similarity: `uel_kernel(pattern_node, sequence_node)`
3. Activation similarity: Normalized relative to `avg_activation`
4. Combined: `max(kernel_sim, activation_sim * 0.5)`

**For blanks:**
- Bind by position (same blank position = same value)
- Contributes 1.0 to similarity when bound

### Pattern Creation vs Matching

**Pattern creation:**
- Sequences repeat twice → create pattern
- Uses similarity to detect blanks (different values = blank)

**Pattern matching:**
- Uses same similarity logic (`uel_kernel`)
- Dynamic threshold (like pattern creation)
- Allows partial matches (like pattern creation)

## Current Status

✅ **Dynamic threshold** - Scales with graph state
✅ **Relative threshold** - Based on `avg_edge_strength` and `avg_activation`
✅ **Partial matches** - Threshold allows 5-90% similarity
✅ **Similarity computation** - Uses `uel_kernel` and activation similarity

**The architecture is correct** - pattern matching now follows the same rules as pattern creation!

