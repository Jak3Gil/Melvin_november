#ifndef MELVIN_H
#define MELVIN_H

#include <stdint.h>
#include <stddef.h>

#define MELVIN_MAGIC 0x4D4C564E // "MLVN"
#define MELVIN_VERSION 2

// Node kinds
#define NODE_KIND_BLANK         0
#define NODE_KIND_DATA          1
#define NODE_KIND_PATTERN_ROOT  2
#define NODE_KIND_CONTROL       3
#define NODE_KIND_TAG           4
#define NODE_KIND_META          5

// Edge flags
#define EDGE_FLAG_ACTIVE        (1 << 0)
#define EDGE_FLAG_SEQ           (1 << 1)
#define EDGE_FLAG_BIND          (1 << 2)
#define EDGE_FLAG_CONTROL       (1 << 3)
#define EDGE_FLAG_ROLE          (1 << 4)
#define EDGE_FLAG_REL           (1 << 5)
#define EDGE_FLAG_CHAN          (1 << 6)
#define EDGE_FLAG_PATTERN       (1 << 7)
#define EDGE_FLAG_MODULE_BYTES  (1 << 8)

// Role flags for pattern edges
#define ROLE_SEQ_FIRST          1
#define ROLE_SEQ_SECOND         2
#define ROLE_SEQ_THIRD          3
#define ROLE_SEQ_FOURTH         4
#define ROLE_COND               5
#define ROLE_THEN               6
#define ROLE_ELSE               7
#define ROLE_BODY               8
#define ROLE_INPUT              9
#define ROLE_OUTPUT             10
#define ROLE_CONTROL            11
#define ROLE_SUGGEST            12
#define ROLE_BLANK              13
#define ROLE_SLOT               14
#define ROLE_LHS                15
#define ROLE_RHS                16
#define ROLE_OP                 17

// Brain Header
typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t tick;
    uint64_t node_cap; // Added for capacity tracking
    uint64_t edge_cap; // Added for capacity tracking
    // Future expansion: pattern offsets, free lists, etc.
    uint8_t  padding[224]; // Pad to 256 bytes or similar alignment if needed
} BrainHeader;

// Node Structure
typedef struct {
    float    a;             // Activation
    float    bias;
    float    decay;
    uint32_t kind;          // DATA, BLANK, PATTERN_ROOT, CONTROL, etc.
    uint32_t flags;
    float    reliability;
    uint32_t success_count;
    uint32_t failure_count;
    uint32_t mc_id;         // 0 = none; >0 = index in MC table
    uint16_t mc_flags;
    uint16_t mc_role;
    float    value;         // General purpose value
} Node;

// Edge Structure
typedef struct {
    uint64_t src;
    uint64_t dst;
    float    w;
    uint32_t flags;
    float    elig;          // Eligibility trace
    uint32_t usage_count;
} Edge;

// Runtime Brain handle (not on disk, just helper)
typedef struct {
    BrainHeader *header;
    Node        *nodes;
    Edge        *edges;
    size_t       mmap_size;
    int          fd;
} Brain;

#endif // MELVIN_H

