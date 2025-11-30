# Activation Flow: What Happens When Melvin Starts

A detailed explanation of where activation comes from and where it goes.

## 🚀 Startup: The First Activation

### Step 1: Main Function Creates Bootstrap Nodes

When `main()` runs, it creates **one special node** with high activation:

```c
// From melvin.c line ~691
uint64_t scaffold_node = alloc_node(&g);
g.nodes[scaffold_node].kind = NODE_KIND_CONTROL;
g.nodes[scaffold_node].mc_id = <mc_process_scaffolds_id>;
g.nodes[scaffold_node].bias = 5.0f;  // High bias = always wants to activate
g.nodes[scaffold_node].a = 1.0f;     // ACTIVATION = 1.0 (fully active!)
```

**This is the ONLY node with activation at startup!**

### Step 2: First Tick Begins

```561:601:melvin.c
void melvin_tick(Brain *g) {
    // Initialize simplicity metrics for this tick
    sm_init(&g_simplicity_metrics);
    
    // 0. External input / MC effects
    ingest_input(g);
    
    // 1. Propagate predictions: compute Â
    propagate_predictions(g);
    
    // 2. Apply environment / finalize actual activations (decay, normalization)
    apply_environment(g);
    
    // 3. Compute local errors e_j = A_j - Â_j (this also accumulates into metrics)
    compute_error(g);
    
    // 4. Update weights & node reliability
    update_edges(g);
    update_nodes_from_error(g);
    
    // 5. Run MC-backed nodes chosen by the graph
    run_mc_nodes(g);
    
    // 6. Compute simplicity metrics
    sm_measure_complexity(g, &g_simplicity_metrics);
    sm_measure_patterns(g, &g_simplicity_metrics);
    sm_compute_objective(&g_simplicity_metrics);
    
    // 7. Inject intrinsic reward into graph
    float intrinsic_reward = sm_reward_from_score(&g_simplicity_metrics);
    melvin_send_intrinsic_reward(g, intrinsic_reward);
    
    // 8. Emit outputs if any
    emit_output(g);
    
    // 9. Debug logging
    log_learning_stats(g);
    sm_log(&g_simplicity_metrics, g->header->tick);
    
    g->header->tick++;
}
```

## 📊 Detailed Flow Per Tick

### Phase 0: `ingest_input(g)` - External Input

```200:203:melvin.c
void ingest_input(Brain *g) {
    // Input ingestion handled by MC nodes or external systems
    // This is a placeholder for any runtime-level input processing
}
```

**Currently empty!** Input comes from MC nodes (see Phase 5).

### Phase 1: `propagate_predictions(g)` - Activation Spreads

```205:234:melvin.c
void propagate_predictions(Brain *g) {
    ensure_buffers(g);
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;

    // 1. Reset predictions to bias
    for(uint64_t i=0; i<n; i++) {
        g_predicted_a[i] = g->nodes[i].bias;
    }

    // 2. Sum weighted inputs from edges
    for(uint64_t i=0; i<e_count; i++) {
        Edge *e = &g->edges[i];
        if (e->src < n && e->dst < n) {
            float input = e->w * g->nodes[e->src].a;
            g_predicted_a[e->dst] += input;
        }
    }

    // 3. Apply nonlinearity and clamp to [0, 1]
    for(uint64_t i=0; i<n; i++) {
        g_predicted_a[i] = sigmoid(g_predicted_a[i]);
        // Clamp to prevent explosion
        if (g_predicted_a[i] < 0.0f) g_predicted_a[i] = 0.0f;
        if (g_predicted_a[i] > 1.0f) g_predicted_a[i] = 1.0f;
    }
    
    // Note: We do NOT modify actual activations here.
    // Actual activations are set by ingest_input, MC nodes, or apply_environment.
}
```

**What happens:**
1. **Start with bias**: Each node's prediction = its `bias` value
   - Scaffold node: bias = 5.0 → prediction starts at 5.0
   - Other nodes: bias = 0.0 → prediction starts at 0.0

2. **Sum weighted inputs**: For each edge `src → dst`:
   - `prediction[dst] += weight * activation[src]`
   - If scaffold node (a=1.0) has edges to other nodes, those nodes get activation!

3. **Apply sigmoid**: Squash predictions to [0, 1] range

**Key Point:** This computes **predicted** activation, not actual activation yet!

### Phase 2: `apply_environment(g)` - Decay Current Activation

```236:247:melvin.c
void apply_environment(Brain *g) {
    // Apply decay and normalization
    uint64_t n = g->header->num_nodes;
    for(uint64_t i=0; i<n; i++) {
        Node *node = &g->nodes[i];
        // Decay activation
        node->a *= (1.0f - node->decay);
        // Clamp to [0, 1]
        if (node->a < 0.0f) node->a = 0.0f;
        if (node->a > 1.0f) node->a = 1.0f;
    }
}
```

**What happens:**
- Current activation decays slightly (unless decay = 0)
- Scaffold node: a=1.0 → might decay to 0.99 (if decay = 0.01)

### Phase 3: `compute_error(g)` - Compare Prediction vs Reality

```261:285:melvin.c
void compute_error(Brain *g) {
    ensure_buffers(g);
    uint64_t n = g->header->num_nodes;
    
    // Reset error accumulation for this tick
    double total_error = 0.0;
    
    for(uint64_t i=0; i<n; i++) {
        float a_actual = g->nodes[i].a;
        float a_pred = g_predicted_a[i];
        float e = a_actual - a_pred;
        // Clamp error to reasonable range
        if (e > 1.0f) e = 1.0f;
        if (e < -1.0f) e = -1.0f;
        g_node_error[i] = e;
        
        // Accumulate into simplicity metrics
        // Assume all nodes contribute to general prediction error
        // Channel-specific tracking would require metadata on nodes
        total_error += fabs(e);
    }
    
    // Accumulate total prediction error (approximate - treating all nodes as general input)
    sm_accumulate_prediction_error(&g_simplicity_metrics, total_error / (double)n, 0);
}
```

**What happens:**
- Error = actual activation - predicted activation
- Scaffold node: a=1.0, predicted=0.99 → error = 0.01 (small)
- Nodes with no activation but high prediction → negative error (surprise!)

### Phase 4: `update_edges(g)` - Learning Happens

```287:321:melvin.c
void update_edges(Brain *g) {
    uint64_t e_count = g->header->num_edges;
    const float eta = 0.001f;  // Learning rate
    const float W_MAX = 10.0f;
    const float lambda = 0.9f; // Eligibility trace decay

    for(uint64_t i=0; i<e_count; i++) {
        Edge *e = &g->edges[i];
        if (e->src >= g->header->num_nodes || e->dst >= g->header->num_nodes) continue;
        
        Node *src = &g->nodes[e->src];
        float a_src = src->a;
        float err_dst = g_node_error[e->dst];
        
        // Simple local learning rule: Δw = η * a_src * err_dst
        float dw = eta * a_src * err_dst;
        
        // NaN guard (check for NaN or infinity)
        if (dw != dw || dw > 1e10f || dw < -1e10f) dw = 0.0f;
        
        e->w += dw;
        
        // Clamp weights
        if (e->w > W_MAX) e->w = W_MAX;
        if (e->w < -W_MAX) e->w = -W_MAX;
        
        // Update eligibility trace
        e->elig = lambda * e->elig + a_src;
        
        // Update usage count when edge contributes
        if (fabsf(a_src) > 0.1f) {
            e->usage_count++;
        }
    }
}
```

**What happens:**
- Weight update: `Δw = learning_rate * source_activation * destination_error`
- If scaffold node (a=1.0) connects to node X with error, the edge weight changes!
- **This is how learning happens!**

### Phase 5: `run_mc_nodes(g)` - MC Functions Execute

```162:176:melvin.c
void run_mc_nodes(Brain *g) {
    for (uint64_t i = 0; i < g->header->num_nodes; ++i) {
        Node *n = &g->nodes[i];
        if (n->mc_id == 0) continue;
        if (n->a < 0.5f) continue;
        if (n->mc_id < MAX_MC_FUNCS) {
            MCEntry *entry = &g_mc_table[n->mc_id];
            if (entry->fn) {
                entry->fn(g, i);
            } else {
                fprintf(stderr, "MC function missing for id %u\n", n->mc_id);
            }
        }
    }
}
```

**What happens:**
- **Scaffold node** has `a = 1.0` and `mc_id = mc_process_scaffolds`
- **Condition met:** `a >= 0.5` → **MC function executes!**
- `mc_process_scaffolds(g, scaffold_node_id)` runs:
  - Scans `scaffolds/` directory
  - Parses `PATTERN_RULE` comments
  - Creates ~1680 new nodes and ~2800 new edges
  - Deletes scaffold files
  - Sets scaffold node activation to 0.0 (deactivates itself)

**This is where the graph EXPLODES with new nodes!**

### Phase 6-7: Simplicity Metrics & Reward

- Computes simplicity score
- Injects intrinsic reward into reward node
- Reward node gets activation spike
- Reward propagates through graph on next tick

### Phase 8: `emit_output(g)` - Output (Currently Empty)

Currently a stub. Output happens via MC nodes (e.g., `mc_chat_out`).

## 🔄 What Happens After First Tick

### Tick 1 (After Scaffolds Processed):

1. **Scaffold node** deactivates (a = 0.0)
2. **~1680 new nodes** exist (from scaffolds)
3. **~2800 new edges** exist (connections within patterns)
4. **Reward node** might have activation (from simplicity reward)
5. **Pattern nodes** exist but have no activation yet

### Tick 2:

1. **Propagation**: Reward node activation spreads through edges
2. **Pattern nodes** might get small activation from reward
3. **MC nodes** check for activation:
   - `mc_parse` node (if created) might activate
   - `mc_chat_in` node (if created) might activate
   - Other MC nodes wait for activation

### Tick 3+:

1. **If `mc_parse` activates:**
   - Scans for C files
   - Parses them
   - Creates function nodes, call edges
   - **More nodes/edges added!**

2. **If `mc_chat_in` activates:**
   - Reads from stdin (non-blocking)
   - Creates word nodes for input
   - Creates sequence edges
   - **More nodes/edges added!**

3. **Activation propagates:**
   - New nodes get activation
   - Activation flows through edges
   - More MC nodes might activate
   - **Cascade of activity!**

## 🎯 Key Points

### Where Activation Comes From:

1. **Bootstrap nodes** (created in `main()`):
   - Scaffold processing node: `a = 1.0` at startup

2. **MC functions** (set activation directly):
   - `mc_chat_in`: Reads stdin, sets word node activations
   - `mc_parse`: Creates function nodes, might activate them
   - `mc_scaffold`: Creates pattern nodes (usually a=0.0 initially)

3. **Propagation** (activation flows through edges):
   - Node A (a=1.0) → Edge (w=2.0) → Node B gets activation
   - Formula: `prediction[B] = bias[B] + sum(weight * activation[source])`

4. **Intrinsic reward** (from simplicity objective):
   - Reward node gets activation spike
   - Propagates through graph

5. **Bias** (nodes want to activate):
   - High bias = node wants to be active
   - Low bias = node stays quiet

### Where Activation Goes:

1. **Through edges** (weighted propagation):
   - Strong edges (high weight) → more activation flows
   - Weak edges (low weight) → less activation flows

2. **To MC nodes** (triggers execution):
   - If `node.a >= 0.5` and `node.mc_id > 0` → MC function runs
   - MC function can create nodes, read input, write output

3. **To learning** (weight updates):
   - High activation + error → edge weight changes
   - This is how the graph learns!

4. **To decay** (fades over time):
   - `a *= (1.0 - decay)`
   - Unless refreshed, activation fades

## 📈 Example: First 10 Ticks

```
Tick 0 (Startup):
  - Scaffold node: a = 1.0
  - All other nodes: a = 0.0
  - Nodes: 1, Edges: 0

Tick 1:
  - Scaffold node: a = 1.0 → mc_process_scaffolds() runs
  - Creates ~1680 nodes, ~2800 edges
  - Scaffold node: a = 0.0 (deactivates)
  - Nodes: 1681, Edges: 2800

Tick 2:
  - Reward node: a = 0.1 (from simplicity reward)
  - Propagation: Reward spreads slightly
  - Nodes: 1681, Edges: 2800

Tick 3:
  - If mc_parse node exists and activates:
    - Scans for C files
    - Parses them
    - Creates ~3050 nodes, ~6100 edges
  - Nodes: 4731, Edges: 8900

Tick 4-10:
  - Activation propagates
  - More MC nodes might activate
  - More nodes/edges created
  - Graph grows!
```

## 💡 Bottom Line

**Initial Input:**
- **ONE node** with activation = 1.0 (scaffold processing node)
- Created in `main()` with high bias

**Activation Flow:**
1. Bootstrap node activates → MC function runs
2. MC function creates nodes/edges
3. New nodes might activate (via bias or propagation)
4. Activation propagates through edges
5. More MC nodes activate → more nodes/edges created
6. **Cascade of growth!**

**The graph is self-organizing:**
- Initial seed (scaffold node) → triggers scaffold processing
- Scaffolds create patterns → patterns can activate
- Patterns trigger MC nodes → MC nodes create more structure
- **Emergent behavior from simple rules!**

