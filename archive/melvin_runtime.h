#ifndef MELVIN_RUNTIME_H
#define MELVIN_RUNTIME_H

#include "melvin_file.h"
#include <stdint.h>

// ========================================================
// Execution Constants
// ========================================================

#define EXEC_THRESHOLD 1.0f   // Activation threshold for code execution

// Function signature for executable code blocks
// All executable nodes must follow this signature
typedef void (*ExecutableCode)(MelvinFile *g, uint64_t node_id);

// ========================================================
// A. Pulse Structure
// ========================================================

typedef struct {
    uint64_t node_id;
    float    strength;
} Pulse;

// ========================================================
// Runtime State
// ========================================================

typedef struct {
    MelvinFile *file;
    
    // Pulse buffers (current and next)
    Pulse *current_pulses;
    Pulse *next_pulses;
    uint64_t current_pulse_count;
    uint64_t next_pulse_count;
    uint64_t pulse_buffer_capacity;
    
    // Temporary buffers for processing
    float *node_accumulator;  // Accumulated pulse strength per node
    
    // Event-driven formation tracking
    int bonds_dirty;           // 1 if bonds changed, triggers formation detection
    uint64_t bond_edge_count;  // Number of bond edges
    uint64_t molecule_count;   // Number of detected molecules
} MelvinRuntime;

// ========================================================
// Physics Functions
// ========================================================

// Initialize runtime
int runtime_init(MelvinRuntime *rt, MelvinFile *file);

// Cleanup runtime
void runtime_cleanup(MelvinRuntime *rt);

// Inject external pulse (from sensors)
void inject_pulse(MelvinRuntime *rt, uint64_t node_id, float strength);

// Core physics tick
void physics_tick(MelvinRuntime *rt);

// Internal physics functions (called by physics_tick)
void propagate_pulses(MelvinRuntime *rt);
void apply_weight_decay(MelvinRuntime *rt);
void strengthen_edges_on_use(MelvinRuntime *rt);
void create_nodes_on_new_input(MelvinRuntime *rt);
void create_edges_on_pulse_flow(MelvinRuntime *rt);
void enforce_energy_budget(MelvinRuntime *rt);
void swap_buffers(MelvinRuntime *rt);
void detect_formations(MelvinRuntime *rt, uint64_t *bond_count, uint64_t *molecule_count);

// Law of Execution: Execute hot executable nodes
void execute_hot_nodes(MelvinRuntime *rt);

#endif // MELVIN_RUNTIME_H

