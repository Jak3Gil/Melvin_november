# Routing Fix Plan

## The Problem

Patterns learn efficiently (255 patterns/ex), but routing chain isn't complete:
1. ✅ Patterns created
2. ✅ Values learned  
3. ✅ Edges to EXEC nodes created
4. ❌ **Value extraction when queries arrive**
5. ❌ **Passing values to EXEC nodes**
6. ❌ **Triggering EXEC execution**

## The Solution (General, Learnable)

### Where Pattern Matching Happens

**In `discover_patterns()`:**
- When `count == 2`: Create new pattern
- When `count > 2`: Pattern already exists - **THIS IS WHERE WE FIX IT**

### The Fix

**When a sequence matches an existing pattern (`count > 2`):**
1. Extract bindings from pattern match (already done in `pattern_matches_sequence`)
2. Extract values from blanks using bindings
3. Check if pattern routes to EXEC node
4. Pass values to EXEC node
5. Trigger execution

**This is general** - works for any pattern, not just '+'

### Implementation Location

**In `discover_patterns()`, when `count > 2`:**

```c
for (uint64_t i = 0; i < g->node_count; i++) {
    if (g->nodes[i].pattern_data_offset > 0) {
        uint32_t bindings[256] = {0};
        if (pattern_matches_sequence(g, (uint32_t)i, sequence, length, bindings)) {
            /* Pattern matches - extract values and route to EXEC */
            extract_and_route_to_exec(g, (uint32_t)i, bindings);
        }
    }
}
```

### New Function: `extract_and_route_to_exec()`

**General mechanism:**
1. Read pattern to find blanks
2. Extract values from bindings for each blank
3. Check if pattern routes to EXEC node
4. Pass values and trigger execution

**This is learnable** - patterns learn which EXEC nodes to route to through edges!

## Why This Works

1. **General**: Works for any pattern, not just '+'
2. **Learnable**: Patterns learn routing through edges
3. **Natural**: Happens during pattern discovery, not hardcoded
4. **Efficient**: Only triggers when patterns actually match

## Next Steps

1. Add `extract_and_route_to_exec()` function
2. Call it in `discover_patterns()` when patterns match
3. Test with "1+1=?" query
4. Verify values are extracted and EXEC executes

