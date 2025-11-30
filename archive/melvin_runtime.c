#define _POSIX_C_SOURCE 200809L

#include "melvin_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

// ========================================================
// Bond and Formation Constants
// ========================================================

#define BOND_WEIGHT_MIN 0.20f
#define BOND_USAGE_MIN  0.01f
#define FIRE_THRESHOLD  0.1f
#define NODE_DECAY_RATE 0.5f   // From swap_buffers() - state *= 0.5f

// Noise parameters (continuous, not tick-based)
#define NOISE_PROB_PER_STEP  0.001f   // per node per step
#define NOISE_ENERGY         0.01f

// ========================================================
// Helper Functions
// ========================================================

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Simple RNG (linear congruential)
static uint64_t next_rng(uint64_t *state) {
    *state = *state * 1103515245ULL + 12345ULL;
    return *state;
}

static float random_float(uint64_t *state) {
    next_rng(state);
    return ((float)(*state & 0x7FFFFFFF)) / 2147483647.0f;
}

// Find node index by ID (returns UINT64_MAX if not found)
static uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id) {
    GraphHeader *gh = file->graph_header;
    NodeDisk *nodes = file->nodes;
    
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (nodes[i].id == node_id) {
            return i;
        }
    }
    return UINT64_MAX;
}

// Check if edge exists between two nodes
static int edge_exists_between(MelvinFile *file, uint64_t src_id, uint64_t dst_id) {
    GraphHeader *gh = file->graph_header;
    EdgeDisk *edges = file->edges;
    
    // Find source node
    uint64_t src_idx = find_node_index_by_id(file, src_id);
    if (src_idx == UINT64_MAX) return 0;
    
    NodeDisk *src_node = &file->nodes[src_idx];
    if (src_node->first_edge_index == UINT64_MAX) return 0;
    
    // Check adjacency list
    uint64_t edge_idx = src_node->first_edge_index;
    while (edge_idx != UINT64_MAX && edge_idx < gh->edge_capacity) {
        EdgeDisk *e = &edges[edge_idx];
        if (e->src_id == UINT64_MAX) break; // Invalid edge
        if (e->src_id == src_id && e->dst_id == dst_id) {
            return 1; // Edge exists
        }
        edge_idx = e->next_out_edge;
    }
    
    return 0;
}

// Find free edge slot (returns UINT64_MAX if no free slot)
static uint64_t find_free_edge_slot(MelvinFile *file) {
    GraphHeader *gh = file->graph_header;
    EdgeDisk *edges = file->edges;
    
    // First try existing slots
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (edges[i].src_id == UINT64_MAX) {
            return i;
        }
    }
    
    // Try beyond num_edges but within capacity
    if (gh->num_edges < gh->edge_capacity) {
        return gh->num_edges;
    }
    
    return UINT64_MAX; // Need to grow
}

// Create edge between two nodes (returns 1 on success, 0 on failure)
static int create_edge_between(MelvinFile *file, uint64_t src_id, uint64_t dst_id, float initial_weight) {
    GraphHeader *gh = file->graph_header;
    NodeDisk *nodes = file->nodes;
    EdgeDisk *edges = file->edges;
    
    // Check if edge already exists
    if (edge_exists_between(file, src_id, dst_id)) {
        return 0; // Already exists
    }
    
    // Find free slot
    uint64_t edge_slot = find_free_edge_slot(file);
    
    // Grow graph if needed
    if (edge_slot == UINT64_MAX) {
        if (grow_graph(file, gh->num_nodes, gh->num_edges + 10) < 0) {
            return 0; // Failed to grow
        }
        edges = file->edges; // Update pointer after growth
        gh = file->graph_header;
        edge_slot = find_free_edge_slot(file);
        if (edge_slot == UINT64_MAX) {
            return 0; // Still no space
        }
    }
    
    // Find source node index
    uint64_t src_idx = find_node_index_by_id(file, src_id);
    if (src_idx == UINT64_MAX) {
        return 0; // Source node not found
    }
    
    NodeDisk *src_node = &nodes[src_idx];
    
    // Create edge
    EdgeDisk *new_edge = &edges[edge_slot];
    new_edge->src_id = src_id;
    new_edge->dst_id = dst_id;
    new_edge->weight = initial_weight;
    new_edge->trace = 0.0f;
    new_edge->age = 0.0f;
    new_edge->pulse_count = 0;
    new_edge->usage = 0.0f;          // Initialize usage
    new_edge->last_energy = 0.0f;    // Initialize last_energy
    new_edge->is_bond = 0;           // Initialize is_bond
    new_edge->next_out_edge = UINT64_MAX;
    
    // Link into source node's adjacency list (insert at head)
    new_edge->next_out_edge = src_node->first_edge_index;
    src_node->first_edge_index = edge_slot;
    src_node->out_degree++;
    
    // Update num_edges if needed
    if (gh->num_edges <= edge_slot) {
        gh->num_edges = edge_slot + 1;
    }
    
    return 1; // Success
}

// ========================================================
// Runtime Initialization
// ========================================================

int runtime_init(MelvinRuntime *rt, MelvinFile *file) {
    if (!rt || !file || !file->map) {
        fprintf(stderr, "[runtime_init] Error: invalid arguments\n");
        return -1;
    }
    
    memset(rt, 0, sizeof(MelvinRuntime));
    rt->file = file;
    
    // Initialize event-driven tracking
    rt->bonds_dirty = 1;  // Start dirty so formations are detected on first tick
    rt->bond_edge_count = 0;
    rt->molecule_count = 0;
    
    // Allocate pulse buffers
    rt->pulse_buffer_capacity = 10000;
    rt->current_pulses = malloc(rt->pulse_buffer_capacity * sizeof(Pulse));
    rt->next_pulses = malloc(rt->pulse_buffer_capacity * sizeof(Pulse));
    
    if (!rt->current_pulses || !rt->next_pulses) {
        fprintf(stderr, "[runtime_init] Error: failed to allocate pulse buffers\n");
        runtime_cleanup(rt);
        return -1;
    }
    
    rt->current_pulse_count = 0;
    rt->next_pulse_count = 0;
    
    // Allocate node accumulator
    GraphHeader *gh = file->graph_header;
    rt->node_accumulator = calloc(gh->node_capacity, sizeof(float));
    if (!rt->node_accumulator) {
        fprintf(stderr, "[runtime_init] Error: failed to allocate node accumulator\n");
        runtime_cleanup(rt);
        return -1;
    }
    
    return 0;
}

void runtime_cleanup(MelvinRuntime *rt) {
    if (!rt) return;
    
    free(rt->current_pulses);
    free(rt->next_pulses);
    free(rt->node_accumulator);
    
    memset(rt, 0, sizeof(MelvinRuntime));
}

// ========================================================
// Inject External Pulse
// ========================================================

void inject_pulse(MelvinRuntime *rt, uint64_t node_id, float strength) {
    if (!rt || strength <= 0.0f) return;
    
    GraphHeader *gh = rt->file->graph_header;
    
    // Check if node exists, create if needed
    NodeDisk *nodes = rt->file->nodes;
    NodeDisk *node = NULL;
    uint64_t node_idx = UINT64_MAX;
    
    // Find node
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (nodes[i].id == node_id) {
            node = &nodes[i];
            node_idx = i;
            break;
        }
    }
    
    // Create node if doesn't exist
    if (!node) {
        if (gh->num_nodes >= gh->node_capacity) {
            // Grow graph
            if (grow_graph(rt->file, gh->num_nodes + 1, gh->num_edges) < 0) {
                fprintf(stderr, "[inject_pulse] Failed to grow graph\n");
                return;
            }
            // Update pointers after growth
            nodes = rt->file->nodes;
            gh = rt->file->graph_header;
        }
        
        node_idx = gh->num_nodes++;
        node = &nodes[node_idx];
        node->id = node_id;
        node->bias = 0.1f;
        node->state = 0.0f;
        node->trace = 0.0f;
        node->first_edge_index = UINT64_MAX; // Initialize!
        node->out_degree = 0;
        node->firing_count = 0;
        node->flags = 0;              // Not executable by default
        node->payload_offset = 0;     // No payload by default
        node->payload_len = 0;        // No payload by default
    }
    
    // Add pulse to current buffer
    if (rt->current_pulse_count >= rt->pulse_buffer_capacity) {
        // Grow buffer
        rt->pulse_buffer_capacity *= 2;
        rt->current_pulses = realloc(rt->current_pulses, rt->pulse_buffer_capacity * sizeof(Pulse));
        rt->next_pulses = realloc(rt->next_pulses, rt->pulse_buffer_capacity * sizeof(Pulse));
        if (!rt->current_pulses || !rt->next_pulses) {
            fprintf(stderr, "[inject_pulse] Failed to grow pulse buffers\n");
            return;
        }
    }
    
    rt->current_pulses[rt->current_pulse_count].node_id = node_id;
    rt->current_pulses[rt->current_pulse_count].strength = strength;
    rt->current_pulse_count++;
    
    gh->total_pulses_emitted++;
}

// ========================================================
// Core Physics Functions
// ========================================================

// 0. apply_noise() - Continuous noise injection (Rule P4)
static void apply_noise(MelvinRuntime *rt) {
    GraphHeader *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    uint64_t *rng_state = &gh->rng_state;
    
    // Apply noise per node, per step (no tick dependency)
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (random_float(rng_state) < NOISE_PROB_PER_STEP) {
            // Add noise pulse to current buffer
            if (rt->current_pulse_count >= rt->pulse_buffer_capacity) {
                rt->pulse_buffer_capacity *= 2;
                rt->current_pulses = realloc(rt->current_pulses, rt->pulse_buffer_capacity * sizeof(Pulse));
                if (!rt->current_pulses) {
                    fprintf(stderr, "[apply_noise] Out of memory\n");
                    return;
                }
            }
            rt->current_pulses[rt->current_pulse_count].node_id = nodes[i].id;
            rt->current_pulses[rt->current_pulse_count].strength = NOISE_ENERGY;
            rt->current_pulse_count++;
            gh->total_pulses_emitted++;
        }
    }
}

// 1. propagate_pulses()
void propagate_pulses(MelvinRuntime *rt) {
    GraphHeader *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    EdgeDisk *edges = rt->file->edges;
    uint64_t *rng_state = &gh->rng_state;
    
    rt->next_pulse_count = 0;
    
    // Clear node accumulator
    memset(rt->node_accumulator, 0, gh->node_capacity * sizeof(float));
    
    // Process each pulse in current buffer
    for (uint64_t p = 0; p < rt->current_pulse_count; p++) {
        Pulse *pulse = &rt->current_pulses[p];
        uint64_t node_id = pulse->node_id;
        float strength = pulse->strength;
        
        // Find node
        uint64_t node_idx = find_node_index_by_id(rt->file, node_id);
        if (node_idx == UINT64_MAX) continue;
        
        NodeDisk *node = &nodes[node_idx];
        
        // Update node state
        node->state += strength;
        node->firing_count++;
        node->trace += strength;
        
        // Try to propagate through edges
        uint64_t edge_index = node->first_edge_index;
        while (edge_index != UINT64_MAX && edge_index < gh->edge_capacity) {
            EdgeDisk *edge = &edges[edge_index];
            
            // Safety check
            if (edge->src_id == UINT64_MAX || edge->src_id != node_id) {
                break; // Invalid edge or end of list
            }
            
            // Law 1: Probability = sigmoid(weight)
            float prob = sigmoid(edge->weight);
            if (random_float(rng_state) < prob) {
                // Propagation successful
                
                // Check buffer capacity
                if (rt->next_pulse_count >= rt->pulse_buffer_capacity) {
                    rt->pulse_buffer_capacity *= 2;
                    rt->next_pulses = realloc(rt->next_pulses, rt->pulse_buffer_capacity * sizeof(Pulse));
                    if (!rt->next_pulses) {
                        fprintf(stderr, "[propagate_pulses] Out of memory\n");
                        break;
                    }
                }
                
                // Create new pulse
                float energy_flowed = strength * 0.8f; // Attenuation
                rt->next_pulses[rt->next_pulse_count].node_id = edge->dst_id;
                rt->next_pulses[rt->next_pulse_count].strength = energy_flowed;
                rt->next_pulse_count++;
                
                // Accumulate in destination node
                uint64_t dst_idx = find_node_index_by_id(rt->file, edge->dst_id);
                if (dst_idx != UINT64_MAX) {
                    rt->node_accumulator[dst_idx] += energy_flowed;
                }
                
                // Track energy flow for bond detection
                edge->last_energy += energy_flowed;
                
                // Update edge trace and pulse count
                edge->trace += 1.0f;
                edge->pulse_count++;
                edge->age += 1.0f;
                
                gh->total_pulses_emitted++;
            }
            
            edge_index = edge->next_out_edge;
        }
    }
}

// 2. apply_weight_decay()
void apply_weight_decay(MelvinRuntime *rt) {
    GraphHeader *gh = rt->file->graph_header;
    EdgeDisk *edges = rt->file->edges;
    float decay = gh->weight_decay;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &edges[i];
        if (e->src_id == UINT64_MAX) continue; // Unused edge
        
        // Track bond state change
        uint8_t old_is_bond = e->is_bond;
        
        // Law 3: Decay weight
        e->weight *= (1.0f - decay);
        e->trace *= 0.9f; // Decay trace
        
        // Natural pruning: remove edges with near-zero weight
        if (e->weight < 0.001f) {
            e->weight = 0.0f;
        }
        
        // Reset bond flags when edges decay (Rule 6)
        if (e->weight < BOND_WEIGHT_MIN || e->usage < BOND_USAGE_MIN) {
            e->is_bond = 0;
        }
        
        // Mark bonds as dirty if bond state changed
        if (e->is_bond != old_is_bond) {
            rt->bonds_dirty = 1;
        }
    }
}

// 3. strengthen_edges_on_use()
void strengthen_edges_on_use(MelvinRuntime *rt) {
    GraphHeader *gh = rt->file->graph_header;
    EdgeDisk *edges = rt->file->edges;
    float learning_rate = gh->learning_rate;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &edges[i];
        if (e->src_id == UINT64_MAX) continue; // Unused edge
        
        // Track bond state change
        uint8_t old_is_bond = e->is_bond;
        
        // Update usage from last_energy (Rule 3)
        e->usage = 0.9f * e->usage + 0.1f * e->last_energy;
        e->last_energy = 0.0f; // Reset accumulator
        
        // Law 4: Plasticity - strengthen based on trace
        if (e->trace > 0.0f) {
            e->weight += learning_rate * e->trace;
            e->trace *= 0.5f; // Reduce trace after strengthening
        }
        
        // Clamp weight to prevent explosion
        if (e->weight > 10.0f) {
            e->weight = 10.0f;
        }
        
        // Mark bonds based on thresholds (Rule 3)
        e->is_bond = (e->weight > BOND_WEIGHT_MIN) && (e->usage > BOND_USAGE_MIN);
        
        // Mark bonds as dirty if bond state changed
        if (e->is_bond != old_is_bond) {
            rt->bonds_dirty = 1;
        }
    }
}

// 4. create_nodes_on_new_input()
void create_nodes_on_new_input(MelvinRuntime *rt) {
    // Nodes are created in inject_pulse() when needed
    (void)rt;
}

// 5. create_edges_on_pulse_flow() - FIXED VERSION
// Condition A: Co-activation rule (bootstraps first edges)
// Condition B: Pulse flow rule (reinforces used paths)
void create_edges_on_pulse_flow(MelvinRuntime *rt) {
    GraphHeader *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    uint64_t *rng_state = &gh->rng_state;
    
    float p_new_coactive = 0.1f; // Probability for co-activation edges
    float p_new_pulse = 0.3f;    // Higher probability for pulse flow edges
    
    // CONDITION A: Co-activation rule
    // Create edges between nodes that fired in the same tick
    if (rt->current_pulse_count >= 2) {
        for (uint64_t i = 0; i < rt->current_pulse_count; i++) {
            uint64_t src_id = rt->current_pulses[i].node_id;
            
            for (uint64_t j = i + 1; j < rt->current_pulse_count; j++) {
                uint64_t dst_id = rt->current_pulses[j].node_id;
                
                if (src_id == dst_id) continue; // No self-loops
                
                // Check if edge exists (only check one direction for now)
                if (!edge_exists_between(rt->file, src_id, dst_id)) {
                    if (random_float(rng_state) < p_new_coactive) {
                        create_edge_between(rt->file, src_id, dst_id, 0.2f);
                    }
                }
            }
        }
    }
    
    // CONDITION B: Pulse flow rule
    // Create edges when pulses would flow from i to j
    // (when a pulse in current buffer leads to a pulse in next buffer)
    for (uint64_t i = 0; i < rt->current_pulse_count; i++) {
        uint64_t src_id = rt->current_pulses[i].node_id;
        
        // Check all nodes that will receive pulses
        for (uint64_t j = 0; j < rt->next_pulse_count; j++) {
            uint64_t dst_id = rt->next_pulses[j].node_id;
            
            if (src_id == dst_id) continue;
            
            // Create edge if doesn't exist
            if (!edge_exists_between(rt->file, src_id, dst_id)) {
                if (random_float(rng_state) < p_new_pulse) {
                    create_edge_between(rt->file, src_id, dst_id, 0.15f);
                }
            }
        }
        
        // Also check nodes with accumulated pulses (from accumulator)
        for (uint64_t n = 0; n < gh->num_nodes; n++) {
            if (rt->node_accumulator[n] > 0.1f && nodes[n].id != src_id) {
                uint64_t dst_id = nodes[n].id;
                if (!edge_exists_between(rt->file, src_id, dst_id)) {
                    if (random_float(rng_state) < p_new_pulse) {
                        create_edge_between(rt->file, src_id, dst_id, 0.15f);
                    }
                }
            }
        }
    }
}

// 6. enforce_energy_budget()
void enforce_energy_budget(MelvinRuntime *rt) {
    GraphHeader *gh = rt->file->graph_header;
    float budget = gh->global_energy_budget;
    
    if (rt->next_pulse_count > (uint64_t)budget) {
        // Scale down pulses
        float scale = budget / (float)rt->next_pulse_count;
        for (uint64_t i = 0; i < rt->next_pulse_count; i++) {
            rt->next_pulses[i].strength *= scale;
        }
    }
}

// 7. swap_buffers()
void swap_buffers(MelvinRuntime *rt) {
    Pulse *tmp = rt->current_pulses;
    rt->current_pulses = rt->next_pulses;
    rt->next_pulses = tmp;
    
    rt->current_pulse_count = rt->next_pulse_count;
    rt->next_pulse_count = 0;
    
    // Apply accumulated pulses to node states
    GraphHeader *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (rt->node_accumulator[i] > 0.0f) {
            nodes[i].state += rt->node_accumulator[i];
            // Add bias contribution
            nodes[i].state += nodes[i].bias * 0.1f;
            // Clamp to prevent explosion
            if (nodes[i].state > 10.0f) {
                nodes[i].state = 10.0f;
            }
            gh->total_pulses_absorbed++;
        }
        
        // Dissipation: decay state
        nodes[i].state *= 0.5f;
        nodes[i].trace *= 0.9f;
    }
    
    // Clear accumulator for next tick
    memset(rt->node_accumulator, 0, gh->node_capacity * sizeof(float));
}

// ========================================================
// Law of Execution: Execute Hot Executable Nodes
// ========================================================

// Universal Execution Rule:
// Whenever a node with EXECUTABLE flag accumulates activation above exec_threshold,
// the bytes in its payload slice are treated as machine code and executed.
void execute_hot_nodes(MelvinRuntime *rt) {
    GraphHeader *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    uint8_t *blob = rt->file->blob;
    
    if (!blob) return; // No blob region available
    
    // Iterate through all nodes
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        NodeDisk *node = &nodes[i];
        
        // Skip unused nodes
        if (node->id == UINT64_MAX) continue;
        
        // Check if node is EXECUTABLE
        int is_executable = (node->flags & NODE_FLAG_EXECUTABLE) != 0;
        
        // Check if node is HOT (activation above threshold)
        int is_hot = node->state > EXEC_THRESHOLD;
        
        // Universal Execution Rule
        if (is_executable && is_hot && node->payload_len > 0) {
            // Validate payload bounds
            if (node->payload_offset + node->payload_len > rt->file->file_header->blob_size) {
                fprintf(stderr, "[execute_hot_nodes] Warning: node %llu payload out of bounds\n",
                        (unsigned long long)node->id);
                continue;
            }
            
            // Get code pointer
            void *code_ptr = blob + node->payload_offset;
            
            // Safety: Check alignment (code should be properly aligned)
            // For now, we'll cast directly - in production, you'd want more safety checks
            
            // Cast payload bytes to function pointer
            // All executable nodes must follow: void fn(MelvinFile *g, uint64_t node_id)
            ExecutableCode fn = (ExecutableCode)code_ptr;
            
            // Execute the code
            // This is the ONLY way Melvin runs code - purely physics-driven
            // The code can do anything: file I/O, network, graph edits, etc.
            // Side effects come from whatever that machine code does
            fn(rt->file, node->id);
            
            // After execution, activation may be modified by the code itself
            // This is part of the "reactions cause more energy" principle
        }
    }
}

// ========================================================
// Formation Detection (Molecules)
// ========================================================

// Detect molecules (bond-connected clusters) - Rule S1, S2
// Event-driven: only updates counts, stores in runtime
void detect_formations(MelvinRuntime *rt, uint64_t *bond_count, uint64_t *molecule_count) {
    GraphHeader *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    EdgeDisk *edges = rt->file->edges;
    
    *bond_count = 0;
    *molecule_count = 0;
    rt->bond_edge_count = 0;
    rt->molecule_count = 0;
    
    if (gh->num_nodes == 0) {
        *bond_count = 0;
        *molecule_count = 0;
        return;
    }
    
    // Count bonds
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (edges[i].src_id != UINT64_MAX && edges[i].is_bond) {
            (*bond_count)++;
        }
    }
    rt->bond_edge_count = *bond_count;
    
    // Build bond adjacency map (which nodes are connected by bonds)
    uint8_t *visited = calloc(gh->num_nodes, sizeof(uint8_t));
    uint8_t *edge_visited = calloc(gh->num_edges, sizeof(uint8_t));
    if (!visited || !edge_visited) {
        free(visited);
        free(edge_visited);
        return;
    }
    
    // BFS to find connected components
    for (uint64_t start = 0; start < gh->num_nodes; start++) {
        if (visited[start]) continue;
        
        // BFS from this node
        uint64_t *queue = malloc(gh->num_nodes * sizeof(uint64_t));
        if (!queue) {
            free(visited);
            free(edge_visited);
            return;
        }
        uint64_t queue_front = 0;
        uint64_t queue_back = 0;
        
        queue[queue_back++] = start;
        visited[start] = 1;
        uint64_t component_size = 0;
        uint64_t bond_edges_in_component = 0;
        uint64_t active_nodes_in_component = 0;
        
        while (queue_front < queue_back) {
            uint64_t node_idx = queue[queue_front++];
            component_size++;
            
            // Check if node is active
            if (nodes[node_idx].state > FIRE_THRESHOLD) {
                active_nodes_in_component++;
            }
            
            // Find all bond-connected neighbors
            uint64_t node_id = nodes[node_idx].id;
            uint64_t edge_idx = nodes[node_idx].first_edge_index;
            
            while (edge_idx != UINT64_MAX && edge_idx < gh->edge_capacity) {
                EdgeDisk *e = &edges[edge_idx];
                if (e->src_id == UINT64_MAX || e->src_id != node_id) break;
                
                if (e->is_bond) {
                    // Count edge only once per component
                    if (!edge_visited[edge_idx]) {
                        bond_edges_in_component++;
                        edge_visited[edge_idx] = 1;
                    }
                    
                    // Find destination node index
                    uint64_t dst_idx = find_node_index_by_id(rt->file, e->dst_id);
                    if (dst_idx != UINT64_MAX && !visited[dst_idx]) {
                        visited[dst_idx] = 1;
                        queue[queue_back++] = dst_idx;
                    }
                }
                
                edge_idx = e->next_out_edge;
            }
        }
        
        free(queue);
        
        // Check if component qualifies as molecule (Rule S1)
        if (component_size >= 3) {
            float internal_density = bond_edges_in_component / (float)component_size;
            float coactive_fraction = (component_size > 0) ? 
                (float)active_nodes_in_component / (float)component_size : 0.0f;
            
            if (internal_density >= 0.3f && coactive_fraction >= 0.5f) {
                (*molecule_count)++;
                // No logging here - that's for test/visualization code
            }
        }
    }
    
    rt->molecule_count = *molecule_count;
    
    free(visited);
    free(edge_visited);
}

// ========================================================
// Main Physics Tick
// ========================================================

void physics_tick(MelvinRuntime *rt) {
    // Apply physics laws in order (no tick-based logic)
    
    // 0. Apply continuous noise (Rule P4)
    apply_noise(rt);
    
    // 1. Apply weight decay (may mark bonds dirty)
    apply_weight_decay(rt);
    
    // 2. Propagate pulses
    propagate_pulses(rt);
    
    // 3. Strengthen edges on use (may mark bonds dirty)
    strengthen_edges_on_use(rt);
    
    // 4. Create edges from pulse flow
    create_edges_on_pulse_flow(rt);
    
    // 5. Enforce energy budget
    enforce_energy_budget(rt);
    
    // 6. Swap buffers (updates node states from accumulated pulses)
    swap_buffers(rt);
    
    // 7. Law of Execution: Execute hot executable nodes
    // This is the ONLY way code runs - purely physics-driven
    // Execution is triggered only by activation flow, not by names or types
    execute_hot_nodes(rt);
    
    // 8. Event-driven formation detection (only if bonds changed)
    if (rt->bonds_dirty) {
        uint64_t bond_count = 0;
        uint64_t molecule_count = 0;
        detect_formations(rt, &bond_count, &molecule_count);
        rt->bonds_dirty = 0;  // Clear dirty flag
    }
    
    // Increment tick counter (for file tracking only, not used in physics)
    rt->file->file_header->tick_counter++;
}
