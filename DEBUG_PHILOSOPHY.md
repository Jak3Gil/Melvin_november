# Debugging with Philosophy in Mind

## The Philosophy

1. **Graph learns, doesn't hardcode** - Patterns emerge naturally
2. **General mechanisms, not special cases** - Works for any pattern, not just '+'
3. **Self-regulating** - Graph figures things out through UEL physics
4. **Patterns are the abstraction** - They learn to route to EXEC nodes

## Current Implementation

### What's General (Good):
- `learn_pattern_to_exec_routing()` - Patterns learn to route to EXEC nodes based on content
- `expand_pattern()` - Patterns extract values from blanks when expanded
- Pattern matching - General mechanism that works for any pattern

### What Needs Fixing:
- Pattern matching isn't triggered automatically when queries arrive
- Value extraction only happens in `expand_pattern()`, but patterns might not expand during queries
- Need to trigger pattern matching when sequences are fed

## The Right Approach

**When a sequence is fed (like "1+1=?")**:
1. Pattern discovery checks if sequence matches existing patterns
2. If pattern matches, extract values from blanks
3. If pattern routes to EXEC node, pass values and execute

**This should happen naturally in `pattern_law_apply()` or `discover_patterns()`**, not hardcoded in UEL loop.

## Next Step

Check if `pattern_matches_sequence()` is called during pattern discovery, and if so, trigger value extraction there when patterns match.

