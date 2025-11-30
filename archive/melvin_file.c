#define _POSIX_C_SOURCE 200809L

#include "melvin_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

// Default initial capacities
#define INITIAL_NODE_CAPACITY 1024
#define INITIAL_EDGE_CAPACITY 4096

// ========================================================
// A. create_new_file()
// ========================================================

int create_new_file(const char *path) {
    // Calculate initial file size
    size_t file_header_size = sizeof(MelvinFileHeader);
    size_t graph_header_size = sizeof(GraphHeader);
    size_t node_size = sizeof(NodeDisk);
    size_t edge_size = sizeof(EdgeDisk);
    size_t code_header_size = sizeof(CodeHeader);
    
    uint64_t node_capacity = INITIAL_NODE_CAPACITY;
    uint64_t edge_capacity = INITIAL_EDGE_CAPACITY;
    
    // Calculate offsets
    uint64_t graph_offset = file_header_size;
    uint64_t nodes_offset = graph_offset + graph_header_size;
    uint64_t edges_offset = nodes_offset + (node_capacity * node_size);
    uint64_t code_offset = edges_offset + (edge_capacity * edge_size);
    uint64_t blob_offset = code_offset + code_header_size;
    uint64_t blob_size = 1024 * 1024; // 1MB initial blob capacity
    uint64_t file_size = blob_offset + blob_size;
    
    // Create and truncate file
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return -1;
    }
    
    if (ftruncate(fd, file_size) < 0) {
        perror("ftruncate");
        close(fd);
        return -1;
    }
    
    // Map file
    void *map = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return -1;
    }
    
    // Zero everything
    memset(map, 0, file_size);
    
    // Write file header
    MelvinFileHeader *fh = (MelvinFileHeader*)map;
    fh->magic = MELVIN_MAGIC;
    fh->version = MELVIN_VERSION;
    fh->file_size = file_size;
    fh->graph_offset = graph_offset;
    fh->graph_size = code_offset - graph_offset;
    fh->code_offset = code_offset;
    fh->code_size = code_header_size;
    fh->blob_offset = blob_offset;
    fh->blob_size = blob_size;
    fh->tick_counter = 0;
    
    // Write graph header
    GraphHeader *gh = (GraphHeader*)((uint8_t*)map + graph_offset);
    gh->num_nodes = 0;
    gh->num_edges = 0;
    gh->node_capacity = node_capacity;
    gh->edge_capacity = edge_capacity;
    gh->nodes_offset = nodes_offset;
    gh->edges_offset = edges_offset;
    gh->learning_rate = 0.001f;
    gh->weight_decay = 0.01f;
    gh->pulse_energy_cost = 0.1f;
    gh->global_energy_budget = 10000.0f;
    gh->total_pulses_emitted = 0;
    gh->total_pulses_absorbed = 0;
    gh->rng_state = 1;
    
    // Initialize code header
    CodeHeader *ch = (CodeHeader*)((uint8_t*)map + code_offset);
    ch->num_blocks = 0;
    ch->blocks_offset = code_offset + sizeof(CodeHeader);
    
    // Initialize all nodes (mark as unused with id = UINT64_MAX)
    NodeDisk *nodes = (NodeDisk*)((uint8_t*)map + nodes_offset);
    for (uint64_t i = 0; i < node_capacity; i++) {
        nodes[i].id = UINT64_MAX;
        nodes[i].first_edge_index = UINT64_MAX;
        nodes[i].flags = 0;
        nodes[i].payload_offset = 0;
        nodes[i].payload_len = 0;
    }
    
    // Initialize all edges (mark as unused with src_id = UINT64_MAX)
    EdgeDisk *edges = (EdgeDisk*)((uint8_t*)map + edges_offset);
    for (uint64_t i = 0; i < edge_capacity; i++) {
        edges[i].src_id = UINT64_MAX;
        edges[i].next_out_edge = UINT64_MAX;
    }
    
    // Sync and unmap
    msync(map, file_size, MS_SYNC);
    munmap(map, file_size);
    close(fd);
    
    printf("[create_file] Created new brain file: %s (size: %llu bytes)\n", 
           path, (unsigned long long)file_size);
    return 0;
}

// ========================================================
// B. load_file()
// ========================================================

int load_file(const char *path, MelvinFile *file) {
    if (!file) {
        fprintf(stderr, "[load_file] Error: file pointer is NULL\n");
        return -1;
    }
    
    memset(file, 0, sizeof(MelvinFile));
    
    // Open file
    file->fd = open(path, O_RDWR);
    if (file->fd < 0) {
        perror("open");
        return -1;
    }
    
    // Get file size
    struct stat st;
    if (fstat(file->fd, &st) < 0) {
        perror("fstat");
        close(file->fd);
        return -1;
    }
    file->map_size = st.st_size;
    
    if (file->map_size < sizeof(MelvinFileHeader)) {
        fprintf(stderr, "[load_file] Error: file too small\n");
        close(file->fd);
        return -1;
    }
    
    // Map file (PROT_EXEC reserved for future code execution)
    // Note: On macOS, PROT_EXEC requires special permissions
    file->map = mmap(NULL, file->map_size, 
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED, file->fd, 0);
    if (file->map == MAP_FAILED) {
        perror("mmap");
        close(file->fd);
        return -1;
    }
    
    // Validate file header
    file->file_header = (MelvinFileHeader*)file->map;
    if (file->file_header->magic != MELVIN_MAGIC) {
        fprintf(stderr, "[load_file] Error: invalid magic number (got 0x%llx, expected 0x%llx)\n",
                (unsigned long long)file->file_header->magic,
                (unsigned long long)MELVIN_MAGIC);
        munmap(file->map, file->map_size);
        close(file->fd);
        return -1;
    }
    
    if (file->file_header->version != MELVIN_VERSION) {
        fprintf(stderr, "[load_file] Warning: version mismatch (got %llu, expected %llu)\n",
                (unsigned long long)file->file_header->version,
                (unsigned long long)MELVIN_VERSION);
    }
    
    // Set graph header pointer
    file->graph_header = (GraphHeader*)((uint8_t*)file->map + file->file_header->graph_offset);
    
    // Set nodes array pointer
    file->nodes = (NodeDisk*)((uint8_t*)file->map + file->graph_header->nodes_offset);
    
    // Set edges array pointer
    file->edges = (EdgeDisk*)((uint8_t*)file->map + file->graph_header->edges_offset);
    
    // Set blob pointer (executable code storage)
    // Handle old files that might not have blob_offset set
    if (file->file_header->blob_offset > 0 && 
        file->file_header->blob_offset < file->map_size) {
        file->blob = (uint8_t*)file->map + file->file_header->blob_offset;
    } else {
        file->blob = NULL; // No blob region available (old file format)
    }
    
    // Set code header pointer
    file->code_header = (CodeHeader*)((uint8_t*)file->map + file->file_header->code_offset);
    
    printf("[load_file] Loaded brain file: %s\n", path);
    printf("  Nodes: %llu/%llu, Edges: %llu/%llu, Tick: %llu\n",
           (unsigned long long)file->graph_header->num_nodes,
           (unsigned long long)file->graph_header->node_capacity,
           (unsigned long long)file->graph_header->num_edges,
           (unsigned long long)file->graph_header->edge_capacity,
           (unsigned long long)file->file_header->tick_counter);
    
    return 0;
}

// ========================================================
// C. grow_graph()
// ========================================================

int grow_graph(MelvinFile *file, uint64_t min_nodes, uint64_t min_edges) {
    if (!file || !file->map) {
        fprintf(stderr, "[grow_graph] Error: file not loaded\n");
        return -1;
    }
    
    GraphHeader *gh = file->graph_header;
    uint64_t old_node_cap = gh->node_capacity;
    uint64_t old_edge_cap = gh->edge_capacity;
    
    // Calculate new capacities (grow by 50% or to minimum needed)
    uint64_t new_node_cap = (min_nodes > old_node_cap * 3 / 2) ? min_nodes : (old_node_cap * 3 / 2);
    uint64_t new_edge_cap = (min_edges > old_edge_cap * 3 / 2) ? min_edges : (old_edge_cap * 3 / 2);
    
    if (new_node_cap <= old_node_cap && new_edge_cap <= old_edge_cap) {
        return 0; // No growth needed
    }
    
    printf("[grow_graph] Growing: nodes %llu->%llu, edges %llu->%llu\n",
           (unsigned long long)old_node_cap,
           (unsigned long long)new_node_cap,
           (unsigned long long)old_edge_cap,
           (unsigned long long)new_edge_cap);
    
    // Calculate new offsets
    size_t node_size = sizeof(NodeDisk);
    size_t edge_size = sizeof(EdgeDisk);
    
    uint64_t new_nodes_offset = gh->nodes_offset;  // Same offset
    uint64_t new_edges_offset = new_nodes_offset + (new_node_cap * node_size);
    uint64_t new_code_offset = new_edges_offset + (new_edge_cap * edge_size);
    
    uint64_t old_graph_size = file->file_header->graph_size;
    uint64_t new_graph_size = new_code_offset - file->file_header->graph_offset;
    uint64_t new_file_size = file->file_header->code_offset - file->file_header->graph_offset + new_graph_size;
    
    // Unmap current mapping
    munmap(file->map, file->map_size);
    
    // Grow file
    if (ftruncate(file->fd, new_file_size) < 0) {
        perror("[grow_graph] ftruncate");
        return -1;
    }
    
    // Remap (PROT_EXEC reserved for future code execution)
    file->map_size = new_file_size;
    file->map = mmap(NULL, file->map_size,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED, file->fd, 0);
    if (file->map == MAP_FAILED) {
        perror("[grow_graph] mmap");
        return -1;
    }
    
    // Update pointers
    file->file_header = (MelvinFileHeader*)file->map;
    file->graph_header = (GraphHeader*)((uint8_t*)file->map + file->file_header->graph_offset);
    file->nodes = (NodeDisk*)((uint8_t*)file->map + new_nodes_offset);
    file->edges = (EdgeDisk*)((uint8_t*)file->map + new_edges_offset);
    file->code_header = (CodeHeader*)((uint8_t*)file->map + file->file_header->code_offset);
    
    // Update blob pointer if available
    if (file->file_header->blob_offset > 0 && 
        file->file_header->blob_offset < file->map_size) {
        file->blob = (uint8_t*)file->map + file->file_header->blob_offset;
    } else {
        file->blob = NULL;
    }
    
    // Update headers
    file->file_header->file_size = new_file_size;
    file->file_header->graph_size = new_graph_size;
    
    gh->node_capacity = new_node_cap;
    gh->edge_capacity = new_edge_cap;
    gh->edges_offset = new_edges_offset;
    
    // Initialize new node space
    for (uint64_t i = old_node_cap; i < new_node_cap; i++) {
        file->nodes[i].id = UINT64_MAX;
        file->nodes[i].first_edge_index = UINT64_MAX;
    }
    
    // Initialize new edge space
    for (uint64_t i = old_edge_cap; i < new_edge_cap; i++) {
        file->edges[i].src_id = UINT64_MAX;
        file->edges[i].next_out_edge = UINT64_MAX;
    }
    
    printf("[grow_graph] Graph grown successfully\n");
    return 0;
}

// ========================================================
// Helper: Unmap and close file
// ========================================================

void close_file(MelvinFile *file) {
    if (!file || !file->map) {
        return;
    }
    
    // Sync before closing
    msync(file->map, file->map_size, MS_SYNC);
    
    munmap(file->map, file->map_size);
    close(file->fd);
    
    memset(file, 0, sizeof(MelvinFile));
}

