# Detailed Routing Chain Analysis

## The Complete Flow

### Step 1: Feed Examples (✅ Working)
```
Examples: "1+2=3", "2+3=5", etc.
→ melvin_feed_byte() called for each byte
→ pattern_law_apply() called
→ Patterns created: [blank0, '+', blank1, '=', blank2]
→ Edges created: '+' → EXEC_ADD
```

### Step 2: Feed Query (❓ Unknown)
```
Query: "1+1=?"
→ melvin_feed_byte() called for each byte: '1', '+', '1', '=', '?'
→ pattern_law_apply() called for each byte
→ Sequence buffer: ['1', '+', '1', '=', '?']
```

### Step 3: Pattern Matching (❌ Likely Failing Here)
**Location:** `pattern_law_apply()` → lines 4118-4243

**What should happen:**
1. For each sequence length (2-10), extract sequence from buffer
2. Build similarity index: find similar nodes for each position
3. Find patterns containing similar nodes
4. Call `pattern_matches_sequence()` to verify match
5. If match, call `extract_and_route_to_exec()`

**Potential issues:**
- **Issue A:** Similarity search might not find patterns
  - Line 4129: `similarity_threshold = (energy * strength) / node_count`
  - If threshold too high, no similar nodes found
  - If no similar nodes, patterns skipped (line 4204)
  
- **Issue B:** Pattern length mismatch
  - Line 4188: `if (length_diff > 3) continue;`
  - Pattern might be different length than query
  
- **Issue C:** Pattern matching might not be called
  - Line 4118: Only runs if `pattern_count > 0`
  - But patterns might exist, just not found in first 1000 nodes

### Step 4: Extract Values (❌ Failing)
**Location:** `extract_and_route_to_exec()` → lines 3635-3715

**What should happen:**
1. For each blank in pattern, get binding: `bindings[blank_pos]`
2. Extract byte sequence from bound node
3. Parse bytes to integer value
4. Store in `extracted_values[]`

**Potential issues:**
- **Issue D:** Bindings not set correctly
  - Line 3639: `if (elem->is_blank != 0 && bindings[elem->value] > 0)`
  - `bindings[elem->value]` - is `elem->value` the blank position?
  - Line 3442: `blank_pos = elem->value` - yes, it is
  - But line 3639 uses `bindings[elem->value]` directly - should use `blank_pos`
  
- **Issue E:** Blank binding check wrong
  - Line 3641: `uint32_t bound_node = bindings[elem->value];`
  - Should be: `uint32_t blank_pos = elem->value; uint32_t bound_node = bindings[blank_pos];`
  - Fixed in line 3643-3645, but might have logic error

- **Issue F:** Byte sequence extraction failing
  - Line 3658-3688: Follows edges to collect multi-digit numbers
  - Edge weight threshold: `avg_edge_strength * 0.5f`
  - If edges too weak, sequence not collected
  - For single digits, should just read `node->byte`

- **Issue G:** Value extraction confidence too high
  - Line 3695: `if (val.value_type == 0 && val.confidence >= confidence_threshold)`
  - `confidence_threshold = avg_activation * 0.1f`
  - If `avg_activation` low, threshold low, should pass
  - But `extract_pattern_value()` might return low confidence

### Step 5: Route to EXEC (❌ Failing)
**Location:** `extract_and_route_to_exec()` → lines 3702-3715

**What should happen:**
1. Find edge from pattern node to EXEC node
2. Call `pass_values_to_exec()` with extracted values
3. Store values in EXEC node's blob
4. Activate EXEC node

**Potential issues:**
- **Issue H:** No edge from pattern to EXEC
  - Line 3704: Searches `pattern_node->first_out` for EXEC node
  - Edge should exist (created by `learn_pattern_to_exec_routing()`)
  - But might not be found if edge list corrupted
  
- **Issue I:** Values not stored correctly
  - Line 3611: `input_offset = exec_node->payload_offset + 256`
  - Values stored at this offset
  - Test checks: `payload_offset + 256` (line 85)
  - Should match, but might be wrong offset

### Step 6: EXEC Activation (❌ Failing)
**Location:** `pass_values_to_exec()` → lines 3617-3625

**What should happen:**
1. Activation boost: `avg_activation * 5.0f`
2. Add to EXEC node: `exec_node->a += activation_boost`
3. Add to propagation queue: `prop_queue_add(g, exec_node_id)`
4. During UEL loop, `melvin_execute_exec_node()` called when threshold exceeded

**Potential issues:**
- **Issue J:** Activation too low
  - Line 3620: `activation_boost = avg_activation * 5.0f`
  - If `avg_activation` very low (e.g., 0.01), boost = 0.05
  - Threshold: `avg_activation * 0.5` = 0.005
  - Should exceed, but might decay before execution
  
- **Issue K:** Execution not triggered
  - Line 2249: `melvin_execute_exec_node(g, node_id)` called in UEL loop
  - Only if `activation >= threshold`
  - Threshold: `avg_activation * 0.5` (line 2978)
  - If activation decays, might not exceed threshold

### Step 7: Result Output (❌ Failing)
**Location:** `melvin_execute_exec_node()` → lines 3050-3070

**What should happen:**
1. Execute machine code
2. Get result
3. Call `convert_result_to_pattern()` to feed result back
4. Result appears in graph

**Potential issues:**
- **Issue L:** Execution not happening
  - If activation doesn't exceed threshold, execution never happens
  - Result never computed

## Root Cause Analysis

**Most Likely Issues (in order):**

1. **Issue D/E:** Bindings not extracted correctly
   - The binding check at line 3639 might be wrong
   - Should check `blank_pos` first, then get binding

2. **Issue A:** Similarity search not finding patterns
   - Threshold might be too high
   - Or patterns not in first 1000 nodes checked

3. **Issue C:** Pattern matching not being called
   - Pattern count check might skip matching
   - Or sequence not in buffer when matching happens

4. **Issue F:** Byte sequence extraction failing
   - Edge weight threshold too high
   - Or edges not strong enough

5. **Issue J/K:** EXEC activation/execution failing
   - Activation might decay before execution
   - Or threshold too high

## Debugging Strategy

1. **Add logging** to verify each step:
   - Is pattern matching called?
   - Are similar nodes found?
   - Are patterns found?
   - Are bindings set?
   - Are values extracted?
   - Is EXEC activated?
   - Is execution triggered?

2. **Check values** at each step:
   - Similarity threshold value
   - Pattern count
   - Binding values
   - Extracted values
   - EXEC activation
   - Execution threshold

3. **Verify assumptions**:
   - Patterns exist and are correct format
   - Edges exist from patterns to EXEC
   - Sequence buffer populated correctly
   - Bindings set correctly

