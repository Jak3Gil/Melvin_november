/*
 * MELVIN SIMPLE INTERFACE
 * 
 * Minimal interface: Just feed bytes to melvin.m, let the graph do the rest.
 * 
 * That's it. No instincts, no exec helpers, no complexity.
 * Just: map brain → feed bytes → tick → read outputs
 */

#ifndef MELVIN_SIMPLE_H
#define MELVIN_SIMPLE_H

#include <stdint.h>
#include <stdbool.h>

// Context (opaque)
typedef struct MelvinSimple MelvinSimple;

// Open melvin.m brain file
// Returns: context on success, NULL on failure
MelvinSimple* melvin_open(const char *brain_path);

// Close and save brain
void melvin_close(MelvinSimple *m);

// Feed a byte to the graph
// channel: which input channel (just a number, graph decides what it means)
// byte: the byte value
void melvin_feed(MelvinSimple *m, uint64_t channel, uint8_t byte);

// Tick the graph (advance physics)
// num_events: how many events to process
void melvin_tick(MelvinSimple *m, uint64_t num_events);

// Read a byte's activation (how "active" is this byte in the graph?)
// Returns: activation value (0.0 = not active, higher = more active)
float melvin_read_byte(MelvinSimple *m, uint8_t byte);

// Get graph stats (read-only)
typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    float avg_activation;
} MelvinStats;

void melvin_stats(MelvinSimple *m, MelvinStats *out);

// Instrumentation: Get edge weight between two bytes (for learning curve tracking)
// Returns: weight if edge exists, 0.0 if not found
float melvin_get_edge_weight(MelvinSimple *m, uint8_t src_byte, uint8_t dst_byte);

// Create new brain file (one-time setup)
int melvin_simple_create_brain(const char *path);

#endif // MELVIN_SIMPLE_H

