/*
 * melvin.h - Pure Binary Brain Loader
 * 
 * .m files are self-contained: graph + machine code (laws) in blob
 * melvin.c is ONLY: mmap, feed bytes, syscall bridge, jump into blob
 * NO physics, NO loops, NO laws in C runtime
 */

#ifndef MELVIN_H
#define MELVIN_H

#include <stdint.h>
#include <stddef.h>

#define MELVIN_MAGIC "MLVN"
#define MELVIN_VERSION 2  /* Bumped for TB-scale: 64-bit offsets, cold_data region */

/* ========================================================================
 * BINARY .M FILE LAYOUT
 * ======================================================================== */

/* On-disk header (exactly 4096 bytes) - TB-scale ready */
typedef struct {
    char     magic[4];              /* "MLVN" */
    uint32_t version;
    uint32_t flags;                 /* Reserved for future use */
    uint64_t file_size;             /* Total file size */
    uint64_t nodes_offset;          /* Node[] start (hot region) */
    uint64_t node_count;            /* Number of hot nodes */
    uint64_t edges_offset;          /* Edge[] start (hot region) */
    uint64_t edge_count;            /* Number of hot edges */
    uint64_t blob_offset;           /* blob[] start (hot region) */
    uint64_t blob_size;             /* Hot blob size */
    uint64_t cold_data_offset;      /* Cold data slab start */
    uint64_t cold_data_size;        /* Cold data slab size */
    uint64_t main_entry_offset;     /* Main entrypoint (relative to blob_offset) */
    uint64_t syscalls_ptr_offset;   /* Where to store syscalls pointer (relative to blob_offset) */
    uint8_t  pad[4096 - (4 + 4 + 4 + 8*12)];  /* Pad to 4096 bytes */
} MelvinHeader;

_Static_assert(sizeof(MelvinHeader) == 4096, "MelvinHeader must be exactly 4096 bytes");

/* Binary node layout */
typedef struct {
    float    a;
    uint8_t  byte;
    uint16_t in_degree;
    uint16_t out_degree;
    uint32_t first_in;
    uint32_t first_out;
} Node;

/* Binary edge layout */
typedef struct {
    uint32_t src;
    uint32_t dst;
    float    w;
    uint32_t next_in;
    uint32_t next_out;
} Edge;

/* In-memory overlay (ONLY loader state, NO physics state) */
typedef struct {
    int           fd;
    void         *map_base;
    size_t        map_size;
    MelvinHeader *hdr;
    Node         *nodes;            /* Hot nodes */
    Edge         *edges;             /* Hot edges */
    uint8_t      *blob;             /* Hot blob */
    uint8_t      *cold_data;        /* Cold data slab (read-only corpus) */
    uint64_t     node_count;        /* Hot node count */
    uint64_t     edge_count;        /* Hot edge count */
    uint64_t     blob_size;         /* Hot blob size */
    uint64_t     cold_data_size;    /* Cold data size */
} Graph;

/* ========================================================================
 * SYSCALL TABLE (for machine code in blob)
 * ======================================================================== */

/* GPU compute request (blob -> host) */
typedef struct {
    const void *kernel_code;      /* GPU kernel code (CUDA PTX, Metal MSL, OpenCL, etc.) */
    size_t kernel_code_len;
    const void *input_data;        /* Input buffer */
    size_t input_data_len;
    void *output_data;             /* Output buffer (host allocates) */
    size_t output_data_len;
    size_t work_dim[3];            /* Grid/thread dimensions */
    const char *kernel_name;       /* Entry point name */
} GPUComputeRequest;

typedef struct {
    void (*sys_write_text)(const uint8_t *bytes, size_t len);
    void (*sys_send_motor_frame)(const uint8_t *frame, size_t len);
    void (*sys_write_file)(const char *path, const uint8_t *data, size_t len);
    int  (*sys_read_file)(const char *path, uint8_t **out_buf, size_t *out_len);
    int  (*sys_run_cc)(const char *src_path, const char *out_path);
    /* GPU: blob can request GPU compute, host handles driver */
    int  (*sys_gpu_compute)(const GPUComputeRequest *req);
} MelvinSyscalls;

/* Helper for blob code to get syscalls pointer */
MelvinSyscalls* melvin_get_syscalls_from_blob(Graph *g);

/* ========================================================================
 * PUBLIC API (PURE LOADER - NO PHYSICS)
 * ======================================================================== */

/* Open/create .m file (mmaps it, does NOT run physics) */
Graph* melvin_open(const char *path, size_t initial_nodes, size_t initial_edges, size_t blob_size);

/* Sync changes to disk */
void melvin_sync(Graph *g);

/* Close and unmap */
void melvin_close(Graph *g);

/* Set syscall table (writes pointer into blob so machine code can find it) */
void melvin_set_syscalls(Graph *g, MelvinSyscalls *syscalls);

/* Feed byte into graph (ONLY writes to mapped .m, NO physics) */
void melvin_feed_byte(Graph *g, uint32_t port_node_id, uint8_t b, float energy);

/* Jump into .m blob at main entrypoint (ONLY way to "run" - blob does everything) */
void melvin_call_entry(Graph *g);

/* Debug helpers (read-only inspection) */
float melvin_get_activation(Graph *g, uint32_t node_id);

/* Cold data access - graph can copy from cold to hot blob */
void melvin_copy_from_cold(Graph *g, uint64_t cold_offset, uint64_t length, uint64_t blob_target_offset);

/* Create a new v2 .m file with hot + cold layout (for corpus packing) */
/* Returns 0 on success, -1 on error */
int melvin_create_v2(const char *path, 
                     uint64_t hot_nodes, 
                     uint64_t hot_edges, 
                     uint64_t hot_blob_bytes, 
                     uint64_t cold_data_bytes);

/* Host syscall initialization (implemented in host_syscalls.c) */
void melvin_init_host_syscalls(MelvinSyscalls *syscalls);

#endif /* MELVIN_H */
