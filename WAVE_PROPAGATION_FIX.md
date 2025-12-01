# Wave Propagation Fix

## The Philosophy

**Wave propagation should handle everything:**
1. Query bytes activate nodes
2. Pattern matching happens during sequence discovery
3. Matching patterns get activated
4. Activation propagates through edges to EXEC nodes
5. EXEC nodes execute when threshold exceeded

## What We Fixed

1. **Pattern matching in `pattern_law_apply`** ✅
   - When sequences are checked, we match against existing patterns
   - Matching patterns get activated and added to propagation queue
   - This is wave propagation - activation flows naturally

2. **EXEC nodes activated through edges** ✅
   - When patterns match and route to EXEC nodes, they activate them through edges
   - EXEC nodes execute when activation exceeds threshold (normal graph physics)

## The Remaining Bug

**Pattern matching doesn't work for queries:**
- Pattern: "1+1=2" (result is data node '2')
- Query: "1+1=?" (result is '?')
- Pattern matching requires exact match, so '?' doesn't match '2'

**The fix:**
- When patterns are created from "1+1=2" and "2+2=4", the result should be a blank (varies)
- Pattern matching should allow '?' to match result blanks
- This is general - works for any pattern with varying results

## Current Status

**Wave propagation architecture is correct:**
- Pattern matching activates pattern nodes ✅
- Activation propagates through edges ✅
- EXEC nodes execute when threshold exceeded ✅

**But pattern matching needs to handle queries:**
- Result position should be blank if it varies
- '?' should match result blanks

The wave propagation will work once pattern matching is fixed!

