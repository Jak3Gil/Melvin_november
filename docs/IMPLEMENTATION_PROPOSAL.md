# Implementation: General Value Extraction Mechanism

## Design Principles

1. **General, not special-cased**: Works for numbers, strings, concepts, etc.
2. **Graph learns mappings**: We provide mechanism, graph learns specifics
3. **Emergent**: Graph can discover novel value types
4. **Minimal changes**: Extends existing pattern system

## Data Structures

### 1. Pattern Value (stored in pattern node or blob)

```c
/* Pattern value - general mechanism for extracting values from patterns */
typedef struct {
    uint32_t value_type;       /* Graph learns: 0=number, 1=string, 2=concept, etc. */
    uint64_t value_data;       /* The actual value (interpreted by type) */
    float confidence;          /* How confident is this value extraction? */
} PatternValue;
```

### 2. EXEC Node Input/Output

```c
/* EXEC node can receive values and return results */
typedef struct {
    uint32_t input_count;      /* Number of inputs */
    uint64_t inputs[8];         /* Input values (max 8 for now) */
    uint64_t result;            /* Output value */
    bool has_result;            /* Has computation completed? */
} ExecIO;
```

### 3. Enhanced Pattern Data

```c
/* Add to PatternData structure */
typedef struct {
    // ... existing fields ...
    uint64_t value_offset;      /* Offset to PatternValue (0 = no value) */
    uint32_t value_type_hint;   /* Graph learns: what type of value this extracts */
} PatternData;
```

## Implementation Steps

### Step 1: Add Value Storage to Pattern Nodes

**File: `src/melvin.h`**

```c
/* Add to Node structure */
typedef struct {
    // ... existing fields ...
    uint64_t pattern_value_offset;  /* If > 0: points to PatternValue in blob */
} Node;
```

### Step 2: General Value Extraction in Pattern Expansion

**File: `src/melvin.c` - Enhance `expand_pattern()`**

```c
/* Extract value from pattern sequence - GENERAL mechanism */
static PatternValue extract_pattern_value(Graph *g, const uint32_t *sequence, 
                                          uint32_t length, uint32_t pattern_node_id) {
    PatternValue value = {0};
    
    if (!g || !sequence || length == 0) return value;
    
    /* Check if pattern node has learned a value mapping */
    Node *pattern_node = &g->nodes[pattern_node_id];
    if (pattern_node->pattern_value_offset > 0) {
        /* Pattern has learned value - read it from blob */
        uint64_t value_offset = pattern_node->pattern_value_offset - g->hdr->blob_offset;
        if (value_offset < g->blob_size) {
            PatternValue *stored_value = (PatternValue *)(g->blob + value_offset);
            value = *stored_value;
        }
    } else {
        /* Pattern hasn't learned value yet - try to infer from sequence */
        /* GENERAL: Works for any sequence type */
        
        /* Check if sequence is all digits (could be a number) */
        bool all_digits = true;
        for (uint32_t i = 0; i < length; i++) {
            uint8_t byte_val = (uint8_t)(sequence[i] & 0xFF);
            if (byte_val < '0' || byte_val > '9') {
                all_digits = false;
                break;
            }
        }
        
        if (all_digits && length <= 10) {
            /* Could be a number - parse it */
            uint64_t num = 0;
            for (uint32_t i = 0; i < length; i++) {
                uint8_t digit = (uint8_t)(sequence[i] & 0xFF) - '0';
                num = num * 10 + digit;
            }
            value.value_type = 0;  /* Number type */
            value.value_data = num;
            value.confidence = 0.5f;  /* Low confidence - graph should learn this */
        } else {
            /* Could be string or concept - store as node sequence */
            value.value_type = 1;  /* String/concept type */
            value.value_data = (uint64_t)sequence[0];  /* First node as identifier */
            value.confidence = 0.3f;  /* Very low confidence */
        }
    }
    
    return value;
}
```

### Step 3: Enhanced Pattern Expansion with Value Extraction

**File: `src/melvin.c` - Modify `expand_pattern()`**

```c
static void expand_pattern(Graph *g, uint32_t pattern_node_id, const uint32_t *bindings) {
    // ... existing code ...
    
    /* NEW: Extract values from pattern if it matches a sequence */
    PatternValue extracted_values[16];  /* Max 16 values per pattern */
    uint32_t value_count = 0;
    
    /* If pattern has blanks, extract values from bindings */
    for (uint32_t i = 0; i < pattern_data->element_count && value_count < 16; i++) {
        PatternElement *elem = &pattern_data->elements[i];
        
        if (elem->is_blank == 1 && bindings) {
            /* Blank element - extract value from binding */
            uint32_t blank_pos = elem->value;
            if (blank_pos < 256 && bindings[blank_pos] > 0) {
                /* Get sequence that matched this blank */
                uint32_t bound_node = bindings[blank_pos];
                
                /* Try to extract value from this node/sequence */
                /* This is where graph learns: which sequences extract which values */
                PatternValue val = extract_pattern_value(g, &bound_node, 1, pattern_node_id);
                if (val.value_data > 0 || val.value_type > 0) {
                    extracted_values[value_count++] = val;
                }
            }
        }
    }
    
    // ... existing expansion code ...
    
    /* NEW: If pattern routes to EXEC node, pass extracted values */
    if (value_count > 0) {
        /* Check if this pattern routes to an EXEC node */
        uint32_t eid = g->nodes[pattern_node_id].first_out;
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            uint32_t dst = g->edges[eid].dst;
            if (dst < g->node_count && g->nodes[dst].payload_offset > 0) {
                /* This is an EXEC node - pass values to it */
                pass_values_to_exec(g, dst, extracted_values, value_count);
                break;
            }
            eid = g->edges[eid].next_out;
        }
    }
}
```

### Step 4: EXEC Node Value Passing

**File: `src/melvin.c` - New function**

```c
/* Pass values to EXEC node - GENERAL mechanism */
static void pass_values_to_exec(Graph *g, uint32_t exec_node_id, 
                                PatternValue *values, uint32_t value_count) {
    if (!g || exec_node_id >= g->node_count || !values || value_count == 0) return;
    
    Node *exec_node = &g->nodes[exec_node_id];
    if (exec_node->payload_offset == 0) return;  /* Not an EXEC node */
    
    /* Store values in EXEC node's context */
    /* For now, store in node's activation/fields (could use blob for complex data) */
    
    /* Extract numeric values (graph learns which values are numbers) */
    uint64_t numeric_inputs[8] = {0};
    uint32_t num_count = 0;
    
    for (uint32_t i = 0; i < value_count && num_count < 8; i++) {
        if (values[i].value_type == 0) {  /* Number type */
            numeric_inputs[num_count++] = values[i].value_data;
        }
    }
    
    if (num_count >= 2) {
        /* Have at least 2 numbers - could be addition */
        /* Graph learns: which EXEC nodes need which inputs */
        
        /* For EXEC_ADD: pass first two numbers */
        /* This is where graph learns routing */
        exec_node->a += 1.0f;  /* Activate EXEC node */
        
        /* Store inputs in node (could use blob for complex data) */
        /* For now, use node fields as temporary storage */
        exec_node->exec_count = num_count;  /* Store count */
        
        /* Trigger execution */
        melvin_execute_exec_node(g, exec_node_id);
    }
}
```

### Step 5: Enhanced EXEC Execution with Values

**File: `src/melvin.c` - Modify `melvin_execute_exec_node()`**

```c
void melvin_execute_exec_node(Graph *g, uint32_t node_id) {
    // ... existing code ...
    
    /* NEW: Get input values from pattern expansion */
    /* Graph learns: which patterns provide which inputs */
    
    /* For EXEC_ADD: expects 2 integer inputs */
    /* Get from pattern expansion context (stored in node) */
    uint64_t input1 = 0, input2 = 0;
    
    /* Extract from node context (simplified - could use blob) */
    if (g->nodes[node_id].exec_count >= 2) {
        /* Values were passed - extract them */
        /* In real implementation, would read from blob or context */
        input1 = /* ... get from context ... */;
        input2 = /* ... get from context ... */;
    }
    
    /* Execute with inputs */
    if (input1 > 0 || input2 > 0) {
        /* Have inputs - execute */
        // ... execute machine code with inputs ...
        uint64_t result = input1 + input2;  /* For ADD */
        
        /* Store result */
        g->nodes[node_id].exec_success_rate = 1.0f;
        
        /* Convert result back to pattern/value */
        /* Graph learns: how to convert results back to sequences */
        convert_result_to_pattern(g, node_id, result);
    }
}
```

### Step 6: Learning Value Mappings

**File: `src/melvin.c` - New function**

```c
/* Learn value mapping from examples - GRAPH LEARNS THIS */
static void learn_value_mapping(Graph *g, uint32_t pattern_node_id, 
                                const uint32_t *sequence, uint32_t length,
                                PatternValue example_value) {
    if (!g || pattern_node_id >= g->node_count) return;
    
    Node *pattern_node = &g->nodes[pattern_node_id];
    
    /* Check if pattern already has a value */
    if (pattern_node->pattern_value_offset == 0) {
        /* No value yet - learn from example */
        
        /* Store value in blob */
        uint64_t value_offset = /* ... allocate in blob ... */;
        PatternValue *stored_value = (PatternValue *)(g->blob + value_offset);
        *stored_value = example_value;
        
        pattern_node->pattern_value_offset = g->hdr->blob_offset + value_offset;
        
        /* Strengthen connection: pattern → value */
        /* Graph learns: this pattern extracts this value */
    } else {
        /* Already has value - strengthen if matches */
        uint64_t value_offset = pattern_node->pattern_value_offset - g->hdr->blob_offset;
        if (value_offset < g->blob_size) {
            PatternValue *stored_value = (PatternValue *)(g->blob + value_offset);
            if (stored_value->value_type == example_value.value_type &&
                stored_value->value_data == example_value.value_data) {
                /* Matches - strengthen confidence */
                stored_value->confidence = fminf(1.0f, stored_value->confidence + 0.1f);
            }
        }
    }
}
```

## Usage: How Graph Learns

### Example 1: Teaching Number Parsing

```c
/* Feed examples - graph learns mappings */
const char *examples[] = {
    "100",  /* Graph learns: "100" → integer 100 */
    "200",  /* Graph learns: "200" → integer 200 */
    "100+200=300"  /* Graph learns: addition pattern */
};

for (int i = 0; i < 3; i++) {
    /* Feed sequence */
    for (size_t j = 0; j < strlen(examples[i]); j++) {
        melvin_feed_byte(g, 0, (uint8_t)examples[i][j], 0.3f);
    }
    
    /* Graph automatically:
     * 1. Discovers pattern
     * 2. Learns value mapping (from context)
     * 3. Routes to EXEC_ADD
     */
}
```

### Example 2: Query "100+100=?"

```c
/* Feed query */
const char *query = "100+100=?";
for (size_t i = 0; i < strlen(query); i++) {
    melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
}

/* Graph automatically:
 * 1. Recognizes pattern "100+100=?"
 * 2. Extracts values: 100, 100
 * 3. Routes to EXEC_ADD with values
 * 4. EXEC_ADD executes: 100 + 100 = 200
 * 5. Result converted back to pattern "200"
 */
```

## Key Features

1. **General**: Works for numbers, strings, concepts
2. **Learned**: Graph learns mappings from examples
3. **Emergent**: Graph can discover novel value types
4. **Minimal**: Extends existing pattern system

## What Graph Learns

- Which patterns extract which values (from examples)
- Which values route to which EXEC nodes (from patterns)
- How to convert results back to sequences (from examples)

## What We Provide

- General value extraction mechanism
- General EXEC I/O mechanism
- General pattern→EXEC bridge

The graph fills in the specifics!

