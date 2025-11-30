# Melvin Physics Layer Specification

**Version:** 1.0  
**Status:** Core System Architecture  
**Purpose:** Defines the pure physics layer that enables emergent intelligence

---

## Core Principle

**Nodes = points. Edges = channels. Activations = pulses. Everything else emerges.**

No types. No roles. No semantics. Only structure and flow.

---

## What Is a Node?

A **node is nothing but:**

```
- ID (unique integer)
- List of outgoing edges
- A tiny bias (float)
- A tiny local state (float)
- A short-term memory trace (for learning)
```

**NO type. NO role. NO meaning. NO label.**

A node is literally **a junction** — like a point in physical space where energy can flow through.

Nodes *only* exist:
* To route pulses
* To form parts of circuits
* To build larger structures

---

## What Is an Edge?

An **edge is nothing but:**

```
- src node ID
- dst node ID
- weight (float)
- decay/eligibility trace (float)
- age counter (uint64)
```

**NO edge types. NO flags. NO semantic categories.**

An edge is a **potential energy channel**.

It exists to:
* Pass pulses
* Strengthen with use
* Decay with disuse

---

## What Is Activation?

Activation is **a pulse event**, zero semantics:

```
(node_id, strength)
```

A pulse is **just a 1 traveling**.

No types. No flags. No payload. Just energy.

Like:
* An electron
* A spike
* A photon
* A packet

It moves → influences → disappears.

---

## Laws of Activation

### **Law 1 — Propagation**

Pulse at node *i* attempts to travel to each neighbor *j* with probability:

```
p = sigmoid(weight_ij)
```

If successful, a new pulse appears at node j.

### **Law 2 — Dissipation**

If a pulse does not propagate, it disappears.

### **Law 3 — Decay**

Weights decay when unused:

```
weight_ij *= (1 - decay_rate)
```

### **Law 4 — Plasticity**

Weights strengthen when pulses successfully propagate:

```
weight_ij += learning_rate * trace
```

### **Law 5 — Energy Constraint**

Total pulses that can propagate per tick is bounded.

This ensures stability.

---

## Why Nodes Form

Nodes form because Melvin encounters **new sensory patterns** or **novel internal patterns**.

Rules:

1. **New sensory bytes → create new nodes as needed**
2. **Novel pulse patterns → create new nodes for new routing needs**
3. **Stabilized circuits → may spawn new shortcut nodes**

This is analogous to:
* Synaptogenesis
* Dendritic sprouting
* Cortical column growth

No types required.

---

## Why Edges Form

Edges form because pulses pass through two nodes in sequence.

If pulses travel i → j:

```
if no edge exists:
    with probability p_new:
        create edge(i→j) with small weight
```

No types. No roles. Just spontaneous bond formation.

---

## Why Does Activation Move?

Because:
* Edges have nonzero weight
* sigmoid(weight) > 0
* Pulses try to propagate
* Decay kills static pulses
* Global energy pushes pulses forward
* Structural reinforcement supports common routes

This is physical motion, not symbolic computation.

---

## Why Does Activation Stop?

Activation stops when:
* Weight too small
* No outgoing edges
* Energy budget empty
* Random propagation fails
* Decay kills the pulse

Again: pure physics.

---

## Why Does the System Generate More Activation Over Time?

Because:
* More edges → more possible propagation opportunities
* More nodes → more junctions for pulses to spread through
* Stabilized circuits → act like amplification motifs
* Reward slightly boosts persistence

But nothing is "typed." Nothing has "meaning." It's all dynamics.

---

## What About Cameras / Audio / Motors?

At this physics layer?

They **don't exist**.

This layer doesn't know:
* Camera
* Audio
* Motors
* Text
* I/O
* Control
* Tasks

Those are **higher layers** (see MC_CONTRACT.md).

The physics layer only sees:

```
external stimulation = incoming pulses
```

From:
* Camera → pulse storm
* Microphone → pulse storm
* Joints → pulse storm
* Motors → pulse storm
* Reward → pulse storm

Everything is just pulses entering the system.

No types needed.

---

## Where Do "Patterns" or "Concepts" Come From?

They emerge as **stabilized subgraphs**.

Not because you defined nodes or edges to have types, but because:

* Certain flows repeat
* Certain circuits stabilize
* Certain clusters react together
* Certain motifs self-reinforce
* Edge weights strengthen into reliable structures

This is exactly how molecules → cells → organisms emerge.

Types are added **above** physics, not inside it.

---

## Implementation Structure

### File Layout

```
melvin.m (brain file):
  - File Header
  - Graph Header
  - Node Array (typeless)
  - Edge Array (typeless)
  - Code Region (machine code blocks)
```

### Physics Tick Sequence

1. **Decay weights** (Law 3)
2. **Propagate pulses** (Law 1)
3. **Strengthen edges** (Law 4)
4. **Create edges from flow** (spontaneous formation)
5. **Enforce energy budget** (Law 5)
6. **Apply pulses** (Law 2 - or they dissipate)

### Key Functions

- `propagate_pulses()` - Law 1 implementation
- `apply_weight_decay()` - Law 3 implementation
- `strengthen_edges_on_use()` - Law 4 implementation
- `enforce_energy_budget()` - Law 5 implementation
- `swap_buffers()` - Law 2 (dissipation) implementation

---

## Parameters

### Graph-Configurable Physics Parameters

- `learning_rate` - Weight strengthening rate (default: 0.001)
- `weight_decay` - Weight decay rate (default: 0.01)
- `pulse_energy_cost` - Energy cost per pulse (default: 0.1)
- `global_energy_budget` - Max pulses per tick (default: 10000)

### Edge Formation

- `p_new` - Probability of creating new edge on pulse flow (default: 0.1)

### Noise Injection

- `noise_rate` - Probability of noise pulse per node per tick (default: 0.001)
- `noise_strength` - Strength of noise pulses (default: 0.01)

---

## Hard Constraints

1. **No types in nodes/edges** - Physics layer is completely typeless
2. **All meaning emerges** - Patterns, concepts, behaviors from structure alone
3. **MC is optional** - Graph can run with zero MC calls
4. **Pure physics only** - No semantic interpretation at this layer

---

## Testing

To verify physics layer correctness:

1. **Type-free check**: Dump graph structure, verify no semantic fields
2. **Emergence check**: Disable MC, verify patterns still form
3. **Noise check**: Disable all inputs, verify internal activity continues
4. **Structure check**: Verify attractors form from pulse statistics alone

---

## Version History

- **v1.0** (Initial): Pure physics layer specification

---

## Related Documents

- `MC_CONTRACT.md` - MC boundary and emergence invariants
- `rules.md` - Detailed implementation rules

