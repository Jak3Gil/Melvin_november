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
    /* Soft structure hints (learnable, can be modified by UEL physics) */
    float    input_propensity;   /* 0.0-1.0: tendency to receive external input */
    float    output_propensity;  /* 0.0-1.0: tendency to produce external output */
    float    memory_propensity;  /* 0.0-1.0: tendency to retain state (slow decay) */
    uint32_t semantic_hint;      /* Port range hint (0-99=input, 100-199=output, 200-255=control) */
    /* Pattern node support */
    uint64_t pattern_data_offset; /* If > 0: blob offset to PatternData structure (pattern node) */
    uint64_t pattern_value_offset; /* If > 0: blob offset to PatternValue (learned value extraction) */
    /* EXEC node support */
    uint64_t payload_offset;      /* If > 0: blob offset to machine code (EXEC node) */
    float    exec_threshold_ratio; /* Threshold as ratio of avg_activation (default 1.0 = 100%) */
    uint32_t exec_count;          /* Number of times this EXEC node has executed */
    float    exec_success_rate;   /* Success rate of executions (0.0-1.0, updated by feedback) */
} Node;

/* Binary edge layout */
typedef struct {
    uint32_t src;
    uint32_t dst;
    float    w;
    uint32_t next_in;
    uint32_t next_out;
} Edge;

/* ========================================================================
 * PATTERN SYSTEM - Global Law: Patterns form from repeated sequences
 * ======================================================================== */

/* Pattern element: either a data node ID or a blank position */
typedef struct {
    uint8_t is_blank;      /* 0 = data node, 1 = blank */
    uint32_t value;        /* If is_blank=0: node ID, if is_blank=1: blank position (pos0, pos1, etc.) */
} PatternElement;

/* Pattern structure (stored in blob, variable size) */
typedef struct {
    uint32_t magic;                    /* Pattern magic: "PATN" */
    uint32_t element_count;            /* Number of elements in pattern */
    uint32_t instance_count;           /* Number of instances that match this pattern */
    float frequency;                   /* How often this pattern occurs */
    float strength;                    /* Pattern strength (from edge weights/usage) */
    uint64_t first_instance_offset;    /* Offset to first instance (or 0 if none) */
    PatternElement elements[];         /* Variable length: pattern elements */
} PatternData;

/* Pattern instance (links pattern to actual sequence) */
typedef struct {
    uint32_t next_instance_offset;     /* Next instance in linked list (0 = end) */
    uint32_t sequence_length;          /* Length of actual sequence */
    uint32_t sequence_nodes[];         /* Variable length: node IDs of actual sequence */
} PatternInstance;

/* Blank node ID: special value indicating a blank in pattern (0xFFFFFFFF) */
#define PATTERN_BLANK_NODE 0xFFFFFFFF

/* Pattern value - general mechanism for extracting values from patterns */
/* Graph learns which patterns extract which values through examples */
typedef struct {
    uint32_t value_type;       /* Graph learns: 0=number, 1=string, 2=concept, etc. */
    uint64_t value_data;       /* The actual value (interpreted by type) */
    float confidence;          /* How confident is this value extraction? (0.0-1.0) */
} PatternValue;

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
    
    /* Event-driven propagation state */
    float        *last_activation;  /* Track last activation for change detection */
    float        *last_message;     /* Track last message for change detection */
    float        avg_chaos;         /* Running average of chaos (relative measure) */
    float        avg_activation;    /* Running average of activation */
    float        avg_edge_strength;  /* Running average of edge strength */
    
    /* Drive mechanism state */
    float        *output_propensity;  /* Output propensity per node (exploration tracking) */
    float        *feedback_correlation; /* Correlation: output → input → chaos reduction */
    float        *prediction_accuracy;  /* Prediction accuracy per node/pattern */
    float        *stored_energy_capacity; /* Accumulated stored energy (importance/capacity - persists) */
    float        avg_output_activity;   /* Running average of output node activity */
    float        avg_feedback_correlation; /* Running average of feedback success */
    float        avg_prediction_accuracy;  /* Running average of prediction accuracy */
    
    /* Dynamic propagation queue (per-graph, sized to node_count) */
    uint32_t     *prop_queue;          /* Dynamic queue array */
    uint64_t     prop_queue_size;      /* Queue size (based on node_count) */
    _Atomic uint32_t prop_queue_head;  /* Queue head (atomic) */
    _Atomic uint32_t prop_queue_tail;  /* Queue tail (atomic) */
    _Atomic uint8_t  *prop_queued;     /* Bitmap: node queued? (atomic array) */
    
    /* Pattern system: sequence tracking and pattern discovery */
    uint32_t     *sequence_buffer;     /* Rolling buffer for recent byte sequence */
    uint64_t     sequence_buffer_size; /* Size of sequence buffer */
    uint64_t     sequence_buffer_pos;  /* Current position in buffer */
    uint64_t     sequence_buffer_full; /* Flag: buffer is full (wraps) */
    uint32_t     *sequence_hash_table; /* Hash table: [hash, count, first_occurrence_offset, ...] */
    uint32_t     *sequence_storage;     /* Storage for first occurrence sequences */
    uint64_t     sequence_hash_size;   /* Size of hash table */
    uint64_t     sequence_storage_size; /* Size of sequence storage */
    uint64_t     sequence_storage_pos;  /* Current position in sequence storage */
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
    /* Cold data: graph can copy from cold_data to blob (self-directed learning) */
    void (*sys_copy_from_cold)(uint64_t cold_offset, uint64_t length, uint64_t blob_target_offset);
    
    /* Pattern generation tools - all return data that becomes graph structure */
    /* LLM: text → text (local Ollama on Jetson) */
    int  (*sys_llm_generate)(const uint8_t *prompt, size_t prompt_len,
                            uint8_t **response, size_t *response_len);
    /* Vision: image bytes → labels/features (local ONNX/PyTorch on Jetson) */
    int  (*sys_vision_identify)(const uint8_t *image_bytes, size_t image_len,
                               uint8_t **labels, size_t *labels_len);
    /* Audio STT: audio bytes → text (local Whisper/Vosk on Jetson) */
    int  (*sys_audio_stt)(const uint8_t *audio_bytes, size_t audio_len,
                         uint8_t **text, size_t *text_len);
    /* Audio TTS: text → audio bytes (local piper/eSpeak on Jetson) */
    int  (*sys_audio_tts)(const uint8_t *text, size_t text_len,
                          uint8_t **audio_bytes, size_t *audio_len);
    
    /* Code compilation: Graph can compile C code to machine code and learn from it */
    /* Compiles C source to machine code, stores in blob, returns entry offset */
    int (*sys_compile_c)(const uint8_t *c_source, size_t source_len,
                         uint64_t *blob_offset, uint64_t *code_size);
    /* Create EXEC node: makes a node executable by pointing it to blob code */
    /* threshold_ratio: relative to avg_activation (1.0 = 100%, 0.5 = 50%, 2.0 = 200%) */
    /* Returns node_id on success, UINT32_MAX on error */
    uint32_t (*sys_create_exec_node)(uint32_t node_id, uint64_t blob_offset, float threshold_ratio);
} MelvinSyscalls;

/* Helper for blob code to get syscalls pointer */
MelvinSyscalls* melvin_get_syscalls_from_blob(Graph *g);

/* Get current graph context (for syscalls that need Graph*) */
Graph* melvin_get_current_graph(void);

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

/* Load patterns from file (data-driven seeding) */
void melvin_load_patterns(Graph *g, const char *pattern_file, float strength);

/* Jump into .m blob at main entrypoint (ONLY way to "run" - blob does everything) */
void melvin_call_entry(Graph *g);

/* Debug helpers (read-only inspection) */
float melvin_get_activation(Graph *g, uint32_t node_id);

/* Cold data access - graph can copy from cold to hot blob */
void melvin_copy_from_cold(Graph *g, uint64_t cold_offset, uint64_t length, uint64_t blob_target_offset);

/* Create EXEC node: set payload_offset and threshold ratio for a node */
/* threshold_ratio: relative to avg_activation (1.0 = 100%, 0.5 = 50%, 2.0 = 200%) */
/* Returns node_id on success, UINT32_MAX on error */
uint32_t melvin_create_exec_node(Graph *g, uint32_t node_id, uint64_t blob_offset, float threshold_ratio);

/* Progress indicators */
void melvin_set_progress_callback(void (*callback)(const char *message, float percent));
void melvin_progress(const char *message, float percent);

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
