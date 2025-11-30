# Melvin's Complete Laws & Rules Reference

**All rules that govern Melvin's graph physics and behavior**

---

## **0. MASTER RULESET — Physical Laws**

### **SECTION 0.1 — Ontology (What Exists)**

Melvin has exactly four primitives:
1. **Nodes** (state)
2. **Edges** (coupling between nodes)
3. **Energy / Activation** (continuous scalar per node)
4. **Events** (discrete updates)

In addition:
5. **Blob** = continuous byte region inside `melvin.m` used ONLY as machine code payload for EXEC nodes

**No new internal object classes are allowed.** All structures must be expressible as nodes, edges, energy, and events.

---

### **SECTION 0.2 — Universal Execution Law (Active Inference)**

**The ONLY way machine code executes:**

1. A node has the `EXECUTABLE` flag
2. Its activation and local context induce a **continuous EXEC propensity** (firing rate) `λ_exec(i)`:
   - `λ_exec(i)` is a smooth, monotonic function of:
     - node activation `a_i`
     - local free energy in its neighborhood
     - exec_cost and global energy/homeostasis signals
   - Higher activation and better expected free-energy reduction → higher `λ_exec(i)`
3. `EV_EXEC_TRIGGER` events are sampled from this rate process (Poisson-like), so EXEC events become **more frequent** as `λ_exec(i)` increases, but **never rely on a hard inequality** like `a_i > threshold`
4. The EXEC handler runs the machine code at that node's blob payload

**Execution MUST:**
- Subtract activation cost (`exec_cost`)
- Convert return value to energy
- Inject that energy back into the graph **via normal events**
- Obey all safety checks & validation

**EXEC cannot bypass graph physics.**

**Free-Energy: EXEC Reward (Active Inference):**
- Before execution: Compute local free energy `E_before`
- Execute machine code
- After short window: Compute `E_after` for same neighborhood
- Reward = `E_before − E_after` (positive if free energy dropped)
- **Stability update**: EXEC nodes that consistently reduce free energy gain stability

**There are NO limits on EXEC call counts.** Poor EXECs are punished only via negative reward and low stability.

---

### **SECTION 0.3 — Energy Laws**

#### **Law 3.1 — Local Conservation**

A node's energy may only change via:
- Incoming weighted edge messages
- Decay
- Externally injected input bytes (data ingestion)
- EXEC return value (converted to energy)
- Reward energy
- Explicit costs (EXEC, structural modification)

#### **Law 3.2 — Global Bound**

Total activation magnitude `Σ |a_i|` is globally bounded via homeostasis:
- If too high → decay increases, gains decrease
- If too low → small noise / gain increases

**No subsystem may inject unlimited energy.**

---

### **SECTION 0.3.1 — No Hard Thresholds Law (Continuous Physics)**

Melvin's physics is **continuous**. There are no hard inequality gates like "if X > threshold then behavior changes discontinuously".

Instead:
- All "threshold-like" values are treated as parameters of **smooth response curves** (e.g. sigmoids, softplus, tanh)
- Changes in behavior are **gradual** as free energy, activation, or usage vary
- Any actual branch in code is treated as an **implementation detail**, not a governing law

Consequences:
- No global or local behavior is defined by a hard comparison `X > T`
- EXEC firing, stability evolution, pattern creation, and pruning all arise from continuous dynamics over time

---

### **SECTION 0.4 — Edge & Message Rules**

**All influence between nodes MUST occur through edges.**
- No secret shortcuts
- No bypassing via C code
- All structural changes must be local

**Edge message passing formula:**
```
m_i = Σ_j (a_j * w_ji)
a_i(t+1) = decay * a_i + tanh(m_i + bias)
```

**Edges store:**
- weight
- usage
- eligibility
- recency

And learning modifies them **only** through rules defined below.

---

### **SECTION 0.5 — Learning Laws (Free-Energy Based)**

All learning follows **free-energy minimization**:

**Core Learning Rule:**
```
Δw_ij = −η · ε_i · a_j
```

where:
- `η` = learning rate (from param node)
- `ε_i = a_i − prediction_i` = prediction error at postsynaptic node i
- `a_j` = presynaptic node activation

**Prediction Update:**
```
prediction_i(t+1) = (1 − α) · prediction_i(t) + α · a_i(t)
```

**Reward Modulation:**
```
Δw_ij += η · λ · reward · eligibility_ij
```

**There are no other update channels.** No global backprop or magic updates. All learning is online, local, and event-driven.

---

### **SECTION 0.6 — Pattern Laws (Purely Free-Energy Based)**

Patterns are NOT metadata. Patterns are **energy routers** that emerge **only when they reduce local free energy**.

#### **Pattern Creation (Continuous Creation Pressure):**

Pattern node creation is driven by a **continuous creation pressure** based on FE reduction:

- For a candidate pattern P over [A, B, C]:
  - `F_before = Σ F_i` for nodes A, B, C (local free energy without pattern)
  - `F_after = F_P + F_C_new` (free energy with pattern P routing A,B → C)
  - `F_i = α * ε_i² + β * a_i² + γ * C_i` (unified free-energy equation per node)
  - `ΔF = F_before − (F_after + creation_cost)`

- The **pattern creation rate** is a smooth, monotonic function of `ΔF`:
  - If `ΔF` is strongly positive (big FE drop), creation pressure is high
  - If `ΔF` is near zero, creation pressure is low but non-negative
  - If `ΔF` is negative (pattern would increase FE), creation pressure is near zero

**There are NO count-based thresholds.** Patterns form purely based on free-energy reduction, not repetition counts.

#### **A pattern node MUST have:**
- Incoming edges from its constituent nodes (A → P, B → P)
- Outgoing edges to predicted successors (P → C)
- Outgoing edge to EXEC template (P → EXEC_TEMPLATE) for action formation

#### **A pattern node MUST:**
- Sit on real energy paths
- Compete for activation with normal nodes
- Affect predictions
- Participate in normal energy propagation (no special-case skipping)

Patterns that do not route energy or fail to reduce error MUST be pruned by structural rules.

---

### **SECTION 0.7 — Structural Evolution Laws (Unified Stability & Free-Energy)**

All structural changes (create/delete nodes, edges, or blob code) are governed by **unified stability and free-energy laws**:

#### **Unified Free-Energy Equation:**

For each node i:
- Activation: `a_i`
- Prediction: `p_i`
- Error: `ε_i = a_i − p_i`
- Structural / compute complexity: `C_i ≥ 0`
- **Local free energy**: `F_i = α · ε_i² + β · a_i² + γ · C_i`
- **Stability**: `S_i ∈ [0,1]` (updated based on F_i and activity)

where:
- `α` = weight for prediction error² (from param node `FE_ALPHA`)
- `β` = weight for activation² (from param node `FE_BETA`)
- `γ` = weight for complexity (from param node `FE_GAMMA`)
- `C_i` summarizes how much "structure" node i participates in

**Complexity is not "size is bad":**
- Large, highly connected regions are allowed if they consistently reduce free energy
- `C_i` penalizes *wasted* structure: nodes and edges that stay busy but do not contribute to prediction error reduction

#### **Stability Update:**
- Moves toward **1.0** when `F_i < threshold_low` AND `|a_i| > activation_min` (low FE + active)
- Moves toward **0.0** when `F_i > threshold_high` (high FE)
- Otherwise, stability drifts slowly (EMA update)

#### **Creation:**
- **Pattern nodes**: Created only when `F_after + creation_cost < F_before − margin` (purely FE-based, no count thresholds)
- **Edges**: Created on co-activation, strengthened via free-energy learning rule `Δw_ij = −η · ε_i · a_j`
- **EXEC nodes**: Created via code-write node when EXEC template is active (no count caps)
- **Code**: Written only when EXEC nodes are active and code-write node is triggered

#### **Stability-Dependent Decay:**
```
effective_decay = base_decay + boost · stability
```
- Low stability (high FE): decay ~0.90 → vanish quickly
- High stability (low FE): decay up to ~0.97 → persist and form building blocks

#### **Pruning (Stability-Based, NO COUNT LIMITS):**

**Node pruning** occurs when:
```
stability < stability_prune_threshold AND
usage < usage_prune_threshold AND
F_i > fe_prune_min
```

**Edge pruning** occurs when:
```
|weight| < edge_weight_prune_min AND
usage < edge_usage_prune_threshold
```

All response-curve parameters are **param nodes**, not hard-coded constants. Behavior is continuous, not threshold-based.

**There are NO hard limits on node/edge counts.** Pruning is purely stability/FE-based.

---

### **SECTION 0.7.1 — Efficiency Competition Law (Relative Circuits)**

Melvin does **not** penalize large graphs by default. Size alone is not bad. What matters is **relative efficiency** for a given job.

**Efficiency Competition Law:**

1. Multiple circuits (paths of nodes and edges) may participate in solving the same job
2. Each circuit accumulates free energy via the nodes it uses
3. If two circuits achieve **similar free-energy reduction** on the same job:
   - The circuit with **lower total Σ F_i** wins (higher stability, stronger edges)
   - The circuit with **higher Σ F_i** loses (lower stability, more pruning)
4. This holds regardless of absolute size

**Implication:**
- The substrate is free to grow very large and complex
- However, when two different structures implement the **same functional mapping**, the one that uses less energy and less unnecessary structure (lower total free energy) dominates

---

### **SECTION 0.8 — Meta-Parameter Laws**

Meta-parameters MUST be represented as **param nodes**.
- Runtime periodically reads param nodes during homeostasis sweeps
- Updates internal physics values
- Writes back to disk
- EXEC nodes can modify param nodes (meta-learning)

**All thresholds are param nodes:**
- Free-energy weights: `FE_ALPHA`, `FE_BETA`, `FE_GAMMA`
- Stability curve parameters
- Pattern creation curve parameters
- Pruning curve parameters
- EXEC curve parameters
- Physics: `DECAY`, `LEARN_RATE`, `EXEC_COST`, etc.

These are **not hard thresholds**. They control the shape and scale of smooth functions (e.g. sigmoids), and all behavior changes continuously as state variables move.

---

### **SECTION 0.9 — Event Laws**

Melvin's time = events.

**Valid events:**
```
EV_INPUT_BYTE
EV_NODE_DELTA
EV_EXEC_TRIGGER
EV_REWARD_ARRIVAL
EV_HOMEOSTASIS_SWEEP
```

**Everything in the system MUST happen as the result of a discrete event.**

No global synchronous update passes.

---

### **SECTION 0.10 — Safety and Validation Laws**

**Validation MUST:**
- Abort EXEC path if NaN/Inf/invalid
- Prevent out-of-bounds writes
- Freeze EXEC if corruption detected
- Never silently fix errors (fail fast)

**Validation MUST NOT:**
- Modify learning
- Change weights
- Change structure

It only enforces invariants and disables unsafe behavior.

---

### **SECTION 0.11 — Unified Flow Law (Closed Loop)**

Melvin is defined by this loop:

1. Inputs → Data nodes (energy injection)
2. Energy propagates → edges (weighted message passing)
3. Activations update → unified stability + free-energy calculation
4. Threshold crossings → EXEC events (no count caps)
5. Machine code executes → returns scalar
6. Return → energy injection
7. Free-energy reduction → reward (active inference)
8. Structure evolves → nodes, edges, blob grow (purely FE-based, no count limits)
9. Stability-based pruning → remove low-stability, high-FE structures
10. Homeostasis → bounding energy
11. Resource scarcity signals → increase costs, trigger pruning (no hard limits)
12. Loop forever

**If any feature does not fit in this loop, it is invalid.**

---

### **SECTION 0.12 — Implementation Constraints**

- No new C-side logic except enforcing laws
- No new "object types" in the graph
- All behavior must emerge from these rules
- EXEC code MUST use energy and edges to affect graph, not direct writes
- All modifications must maintain `.m` file binary integrity
- **NO HARD-CODED LIMITS**: Any "MAX_NODES", "MAX_EDGES", "MAX_PATTERNS" are implementation details for storage allocation only, not behavioral limits
- **Resource exhaustion**: When allocation fails (OOM, disk full), signal scarcity through param nodes and continue with existing structure. Do not crash or enforce hard limits
- **All response-curve parameters are param nodes**: No magic numbers for pattern creation, pruning, stability, or EXEC triggering

---

## **Additional Laws (From Code Comments)**

### **Unbounded Substrate Rules**

**Rule U1** — The mind is a single graph that can span:
- Local RAM (fast region)
- Memory-mapped files on SSD/NVMe (slower region)
- Remote nodes over network links (distributed regions)
- Any other storage medium we can read/write as bytes

All of these are treated as **one graph**, possibly with different latencies.

**Rule U2** — The runner does nothing except:
- Map storage regions into addressable memory
- Apply energy dynamics (message passing, nonlinearity, homeostasis)
- Execute hot EXECUTABLE nodes (jump into bytes when activation > threshold)
- Expose syscalls
- Sync storage
- **No logic. No interpretation. No policies.**

---

### **Object Rules**

**Rule N1** — A node is a slot that can hold energy.
- Has: ID, energy value, memory (fatigue, bias, trace), list of outgoing edge indices
- Nodes have NO TYPE. A node can mean anything depending on structure.

**Rule E1** — An edge is a directed connection between two nodes.
- Has: source node, destination node, weight, trace
- Edges have NO TYPE. They only carry energy.

**Rule B1** — Any byte region in `melvin.m` can be data or machine code.
- There is no difference. The graph decides what bytes mean.

---

### **Energy Dynamics Rules**

**Rule P1** — Energy flows via message passing.
- For each node i: `m_i = Σ_j (w_ji * a_j)`
- Messages accumulate from all incoming edges weighted by source activations.

**Rule P2** — Energy update with decay + nonlinearity.
- `a_i(t+1) = (1-α)*a_i(t) + α*f(m_i - decay_i + noise_i)`
- `f` is a bounded nonlinearity (tanh-like)
- `decay_i` keeps energy from blowing up
- `noise_i` injects randomness so dormant regions can wake up

**Rule P3** — Homeostasis keeps activity in healthy band.
- If average activation too low → small boosts
- If too high → increased decay
- Goal: maintain rich, structured, non-saturated activity

**Rule P4** — Background noise occasionally creates tiny activations.
- This prevents total silence.

---

### **Node Dynamics Rules**

**Rule ND1** — Nodes have activation (energy) that updates via message passing.
- Activation is a scalar value that represents energy at that node.

**Rule ND2** — Nodes track prediction and prediction error.
- Each node maintains a prediction of its next activation.
- Prediction error `ε = |actual - predicted|` drives learning.

**Rule ND3** — Nodes accumulate reward signals.
- Reward modulates learning: `Δw_ij ∝ (– ε + λ * reward) * e_ij`
- Circuits that lead to predictable, rewarding futures strengthen.

**Rule ND4** — Nodes have energy cost (metabolic cost).
- Being active has a cost proportional to activation magnitude.
- Circuits that maintain prediction and reward with minimal energy are favored.

---

### **Edge Dynamics & Learning Rules**

**Rule ED1** — Edges have weights that determine message strength.
- `m_i += w_ji * a_j` for each incoming edge j→i

**Rule ED2** — Edges maintain eligibility traces for learning.
- Eligibility tracks recent co-activation of source and destination.

**Rule ED3** — Learning driven by prediction error + reward.
- `Δw_ij ∝ (– ε + λ * reward) * e_ij`
- If edge contributed to good prediction (small ε) → weight increases
- If edge contributed to bad prediction (large ε) → weight decreases
- Reward scales learning: edges leading to rewarding outcomes strengthen

**Rule ED4** — Edges decay if not used.
- Slow decay over time prevents weight explosion.

**Rule ED5** — New edges appear between co-active nodes.
- If two nodes are active simultaneously, a new edge may form.

**Rule ED6** — Bad edges disappear over time.
- Edges with near-zero weight are effectively removed.

---

### **Byte Ingestion Rules**

**Rule B1** — External data becomes energy splashes over DATA nodes.
- Each byte value gets a DATA node (or finds existing one)
- Activation burst: `a_{N[b]} += input_energy`

**Rule B2** — Sequential edges (SEQ) connect bytes in order.
- Previous byte → current byte in same channel

**Rule B3** — Channel edges (CHAN) connect channel to data.
- Channel node CH_C → DATA node N[b]

All external data (text, C files, machine code, images, sensor readings, motor feedback, network packets) is reduced to: **energy splashes over DATA nodes plus SEQ/CHAN edges.**

---

### **Structure Formation Rules**

**Rule S1** — Stable loops of nodes become molecules.
- If a set of nodes repeatedly send energy in a loop, they become a stable circuit.

**Rule S2** — Molecules combine into larger motifs.
- Two circuits that often activate together link and form a bigger structure.

**Rule S3** — Motifs form modules.
- Collections of motifs that frequently co-activate become a subsystem.
- Structure is emergent, not defined.

---

### **Universal Execution Rules**

**Rule X1** — Machine code is just another kind of payload in `blob[]`.
- Each node may optionally point at executable payload:
  - `payload_offset(i)` — where bytes start in `blob[]`
  - `payload_len(i)` — length of that slice
  - `flags(i)` — includes bit `EXECUTABLE`

**Rule X2** — Execution Law (CONTINUOUS, NO THRESHOLD):
- For every node i with `EXECUTABLE(i)` set:
  - Compute exec_propensity `λ(i) = sigmoid(k * (activation(i) - center))`
  - Execution **PROBABILITY** is proportional to `λ(i)` (no hard threshold)
  - When firing occurs, treat `blob[payload_offset(i) …]` as MACHINE CODE:
    ```c
    void (*fn)(MelvinFile *g, uint64_t self_id) = 
        (void *)(blob_base + payload_offset(i));
    fn(g, i);  // CPU executes these bytes directly
    ```
- Higher activation = higher firing probability, but **NO hard cutoff**.

**Rule X3** — Code running in `blob[]` (machine code) may:
- Execute directly on CPU/GPU (it IS machine code)
- Perform syscalls (read camera, write motor, etc.)
- Read/write any byte in the graph
- **MODIFY ITS OWN CODE** (self-modifying machine code)
- **WRITE NEW MACHINE CODE** into the blob (code that writes code)
- Modify nodes and edges
- Create new EXECUTABLE nodes pointing to new machine code

**But:** All effects must return **only as energy** injected via events. Code cannot bypass graph physics.

---

## **NEW LAWS (Recently Added)**

### **Global Gravity Law**

**Physics:** Data sculpts the field; energy follows the field.
Every node exerts influence on every other node through learned connectivity.
Activation moves along the gradient of lowest expected prediction error,
and the gradient is reshaped by edge/pattern updates.

**Mechanism:**
1. Compute "mass" for each node (data-driven, depends on learned structure)
2. Compute global field at each node from all massive nodes
3. Add global field term to local message passing

**Mass components** (data-driven):
- Recent activation (smoothed)
- Node degree (in/out edges)
- Edge weight sum (total incoming/outgoing weight)
- Pattern strength (if node is linked to patterns)

**Global field:** `F_i = λ * Σ_j (mass_j * K(i, j))`
- `λ` = gravity_lambda (hyperparameter, very small at first)
- `K(i, j)` = influence kernel (channel similarity, graph distance, etc.)
- Only massive nodes (mass > threshold or top-K) contribute

**The field is continuously reshaped by learning:**
- Edge weight updates → change node degrees → change mass
- Pattern updates → change pattern strength → change mass
- Therefore the global field evolves with the graph

---

### **Instinct System Laws**

**Instincts are global, slowly-varying fields that bias Melvin toward:**
- Not dying (survival: avoid corruption/runaway)
- Exploring but not drowning in noise (novelty)
- Building coherent, compressible structure (coherence)
- Staying in safe energy regime (energy_safe)

**Instinct Update Rules:**
- Instincts updated via leaky integrators based on global stats
- High surprise → increase novelty, decrease coherence
- Very low surprise & low novelty → increase novelty
- Near-corruption/NaN/extreme activations → boost survival, reduce novelty
- Wildly high/low average activation → adjust energy_safe

**Instinct Application:**
- **Survival**: Extra decay when system is unstable (dampen updates)
- **Novelty**: Slight bias toward under-used regions (low mass)
- **Coherence**: Slight bias for nodes in strong patterns (low error)
- **Energy_safe**: Global gain correction to keep activations in target band

**Instinct-Scaled Learning:**
- **Coherence**: Boost learning (up to +20%) for coherent patterns (low error)
- **Novelty**: Slight boost (+10%) when novelty is high (allows structural changes)
- **Survival**: Reduce learning (up to -30%) when survival is high (conservative mode)

All instincts ∈ [0,1], clamped and bounded. Instincts tilt the landscape, not override it.

---

## **Summary: Core Principles**

1. **Continuous Physics**: No hard thresholds, all behavior is smooth and continuous
2. **Local Rules**: All updates are local (nodes/edges), no global optimization
3. **Energy-Based**: Everything flows through energy dynamics
4. **Event-Driven**: All changes happen via discrete events
5. **Unbounded Growth**: No hard limits on graph size (only resource constraints)
6. **Free-Energy Minimization**: Learning and structure formation driven by FE reduction
7. **Self-Modification**: EXEC nodes can write machine code, modify structure
8. **No Special Cases**: All nodes/edges treated equally, no privileged types
9. **Data-Driven**: All behavior emerges from data patterns, not hard-coded logic
10. **Stability-Based Evolution**: Structure persists based on free-energy reduction, not counts

---

**These are the complete laws that govern Melvin's graph. All code must comply with these rules.**

