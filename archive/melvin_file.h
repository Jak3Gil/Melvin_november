#ifndef MELVIN_FILE_H
#define MELVIN_FILE_H

#include <stdint.h>
#include <stddef.h>

#define MELVIN_MAGIC   0x4D4C564EULL  // "MLVN"
#define MELVIN_VERSION 1ULL

// ========================================================
// A. File Header
// ========================================================

typedef struct {
    uint64_t magic;
    uint64_t version;
    uint64_t file_size;
    uint64_t graph_offset;
    uint64_t graph_size;
    uint64_t code_offset;
    uint64_t code_size;
    uint64_t blob_offset;      // Offset to executable blob region
    uint64_t blob_size;        // Size of blob region
    uint64_t tick_counter;
    uint8_t  reserved[256 - 8*9];  // Pad to 256 bytes (7 fields -> 9 fields)
} MelvinFileHeader;

// ========================================================
// B. Graph Header
// ========================================================

typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t node_capacity;
    uint64_t edge_capacity;
    uint64_t nodes_offset;
    uint64_t edges_offset;
    float learning_rate;
    float weight_decay;
    float pulse_energy_cost;
    float global_energy_budget;
    uint64_t total_pulses_emitted;
    uint64_t total_pulses_absorbed;
    uint64_t rng_state;
    uint8_t reserved[128 - (8*9 + 4*4)];  // Pad to 128 bytes
} GraphHeader;

// ========================================================
// C. Node Representation (TYPELESS)
// ========================================================

// Node flags
#define NODE_FLAG_EXECUTABLE (1U << 0)  // Node's payload is executable code

typedef struct {
    uint64_t id;
    float bias;
    float state;              // Activation/energy (used for execution threshold)
    float trace;
    uint64_t first_edge_index;  // Index into edges array (UINT64_MAX = none)
    uint32_t out_degree;
    uint32_t firing_count;
    uint32_t flags;           // Flags (EXECUTABLE bit, etc.)
    uint64_t payload_offset;  // Offset into blob[] for executable code
    uint32_t payload_len;     // Length of payload slice
    uint8_t reserved[4];      // Pad to 64 bytes
} NodeDisk;

// ========================================================
// D. Edge Representation (TYPELESS)
// ========================================================

typedef struct {
    uint64_t src_id;
    uint64_t dst_id;
    float weight;
    float trace;
    float age;
    uint64_t next_out_edge;  // Index to next edge in src's outgoing list (UINT64_MAX = end)
    uint32_t pulse_count;
    float usage;          // moving average of energy that flowed over this edge
    float last_energy;    // temp accumulator for this tick
    uint8_t is_bond;      // 1 if edge qualifies as bond
    uint8_t reserved[3];  // Pad to 64 bytes (was 12, now 3 after adding float + uint8_t)
} EdgeDisk;

// ========================================================
// E. Code Region (for future machine code blocks)
// ========================================================

typedef struct {
    uint64_t num_blocks;
    uint64_t blocks_offset;
    uint8_t reserved[32];  // Pad to 48 bytes
} CodeHeader;

typedef struct {
    uint64_t id;
    uint64_t offset;
    uint64_t size;
    uint64_t entry_offset;
    uint64_t call_count;
    uint64_t last_called_tick;
    uint8_t reserved[32];  // Pad to 64 bytes
} CodeBlockHeader;

// ========================================================
// File Handle Structure
// ========================================================

typedef struct {
    int fd;
    void *map;
    size_t map_size;
    MelvinFileHeader *file_header;
    GraphHeader *graph_header;
    NodeDisk *nodes;
    EdgeDisk *edges;
    CodeHeader *code_header;
    uint8_t *blob;            // Pointer to executable blob region
} MelvinFile;

// ========================================================
// File Operations
// ========================================================

int create_new_file(const char *path);
int load_file(const char *path, MelvinFile *file);
int grow_graph(MelvinFile *file, uint64_t min_nodes, uint64_t min_edges);
void close_file(MelvinFile *file);

#endif // MELVIN_FILE_H

