/*
 * melvin.c - Binary Brain Loader + UEL Physics
 * 
 * Does:
 *   1. mmap .m file
 *   2. feed bytes (write to .m)
 *   3. expose syscalls
 *   4. run UEL physics (embedded in this file)
 * 
 * UEL physics is embedded directly in melvin.c.
 * Blob is for future self-modification, but UEL runs here now.
 */

#include "melvin.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>

/* ========================================================================
 * OPEN/CREATE .M FILE
 * ======================================================================== */

/* Forward declare */
static Graph* melvin_open_with_cold(const char *path, size_t initial_nodes, size_t initial_edges, 
                                     size_t blob_size, size_t cold_data_size);

Graph* melvin_open(const char *path, size_t initial_nodes, size_t initial_edges, size_t blob_size) {
    return melvin_open_with_cold(path, initial_nodes, initial_edges, blob_size, 0);
}

static Graph* melvin_open_with_cold(const char *path, size_t initial_nodes, size_t initial_edges, 
                                     size_t blob_size, size_t cold_data_size) {
    if (!path) return NULL;
    
    Graph *g = calloc(1, sizeof(Graph));
    if (!g) return NULL;
    
    /* Calculate layout - all offsets 64-bit */
    uint64_t header_size = sizeof(MelvinHeader);
    uint64_t nodes_size = (uint64_t)initial_nodes * sizeof(Node);
    uint64_t edges_size = (uint64_t)initial_edges * sizeof(Edge);
    uint64_t blob_size_u64 = (uint64_t)blob_size;
    uint64_t cold_size_u64 = (uint64_t)cold_data_size;
    
    uint64_t off = header_size;
    uint64_t nodes_offset = off;
    off += nodes_size;
    uint64_t edges_offset = off;
    off += edges_size;
    uint64_t blob_offset = off;
    off += blob_size_u64;
    uint64_t cold_data_offset = off;
    off += cold_size_u64;
    uint64_t total_size = off;
    
    /* Round up to page size */
    size_t page_size = getpagesize();
    total_size = (total_size + page_size - 1) & ~(page_size - 1);
    
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        free(g);
        return NULL;
    }
    
    struct stat st;
    fstat(fd, &st);
    bool is_new = (st.st_size == 0);
    
    if (is_new) {
        /* Create new file */
        if (ftruncate(fd, total_size) < 0) {
            close(fd);
            free(g);
            return NULL;
        }
        
        /* mmap */
        void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            close(fd);
            free(g);
            return NULL;
        }
        
        /* Initialize header */
        MelvinHeader *hdr = (MelvinHeader *)map;
        memcpy(hdr->magic, MELVIN_MAGIC, 4);
        hdr->version = MELVIN_VERSION;
        hdr->flags = 0;
        hdr->file_size = total_size;
        hdr->nodes_offset = nodes_offset;
        hdr->node_count = initial_nodes;
        hdr->edges_offset = edges_offset;
        hdr->edge_count = 0;
        hdr->blob_offset = blob_offset;
        hdr->blob_size = blob_size_u64;
        hdr->cold_data_offset = cold_data_offset;
        hdr->cold_data_size = cold_size_u64;
        hdr->main_entry_offset = 0;  /* Set by tool that seeds blob */
        hdr->syscalls_ptr_offset = 0; /* Set by tool */
        
        /* Zero hot regions */
        memset((char *)map + hdr->nodes_offset, 0, (size_t)nodes_size);
        memset((char *)map + hdr->edges_offset, 0, (size_t)edges_size);
        memset((char *)map + hdr->blob_offset, 0, (size_t)blob_size_u64);
        /* Cold data left as-is (will be filled by corpus loader) */
        
        /* Initialize data nodes (0-255) - just structure, no physics */
        Node *nodes = (Node *)((char *)map + hdr->nodes_offset);
        for (int i = 0; i < 256 && i < (int)initial_nodes; i++) {
            nodes[i].byte = (uint8_t)i;
            nodes[i].a = 0.0f;
            nodes[i].first_in = UINT32_MAX;
            nodes[i].first_out = UINT32_MAX;
            nodes[i].in_degree = 0;
            nodes[i].out_degree = 0;
        }
        
        g->fd = fd;
        g->map_base = map;
        g->map_size = (size_t)total_size;
        g->hdr = hdr;
        g->nodes = nodes;
        g->edges = (Edge *)((char *)map + hdr->edges_offset);
        g->blob = (uint8_t *)((char *)map + hdr->blob_offset);
        g->cold_data = (hdr->cold_data_size > 0) ? 
                       (uint8_t *)((char *)map + hdr->cold_data_offset) : NULL;
        g->node_count = hdr->node_count;
        g->edge_count = hdr->edge_count;
        g->blob_size = hdr->blob_size;
        g->cold_data_size = hdr->cold_data_size;
        
    } else {
        /* Open existing file */
        void *map = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            close(fd);
            free(g);
            return NULL;
        }
        
        MelvinHeader *hdr = (MelvinHeader *)map;
        
        /* Validate */
        if (memcmp(hdr->magic, MELVIN_MAGIC, 4) != 0) {
            munmap(map, st.st_size);
            close(fd);
            free(g);
            return NULL;
        }
        
        /* Version check: v1 files have different header layout */
        if (hdr->version < 1 || hdr->version > MELVIN_VERSION) {
            munmap(map, st.st_size);
            close(fd);
            free(g);
            return NULL;
        }
        
        /* Handle v1 files: different header layout */
        if (hdr->version == 1) {
            /* v1 layout (on disk):
             *   offset 0-3:   magic[4]
             *   offset 4-7:   version[4] = 1
             *   offset 8-11:  reserved[4]
             *   offset 12-19: node_count[8]
             *   offset 20-27: edge_count[8]
             *   offset 28-35: nodes_offset[8]
             *   offset 36-43: edges_offset[8]
             *   offset 44-51: blob_offset[8]
             *   offset 52-59: blob_size[8]
             *   offset 60-67: file_size[8]
             *   offset 68-75: main_entry_offset[8]
             *   offset 76-83: syscalls_ptr_offset[8]
             * 
             * v2 layout (in memory):
             *   offset 0-3:   magic[4]
             *   offset 4-7:   version[4] = 2
             *   offset 8-11:  flags[4]
             *   offset 12-19: file_size[8]
             *   offset 20-27: nodes_offset[8]
             *   offset 28-35: node_count[8]
             *   offset 36-43: edges_offset[8]
             *   offset 44-51: edge_count[8]
             *   offset 52-59: blob_offset[8]
             *   offset 60-67: blob_size[8]
             *   offset 68-75: cold_data_offset[8]
             *   offset 76-83: cold_data_size[8]
             *   offset 84-91: main_entry_offset[8]
             *   offset 92-99: syscalls_ptr_offset[8]
             * 
             * Read v1 layout from disk and map to v2 struct */
            uint8_t *p = (uint8_t *)hdr;
            uint64_t v1_node_count, v1_edge_count, v1_nodes_offset, v1_edges_offset;
            uint64_t v1_blob_offset, v1_blob_size, v1_file_size, v1_main_entry, v1_syscalls_ptr;
            
            memcpy(&v1_node_count, p + 12, 8);
            memcpy(&v1_edge_count, p + 20, 8);
            memcpy(&v1_nodes_offset, p + 28, 8);
            memcpy(&v1_edges_offset, p + 36, 8);
            memcpy(&v1_blob_offset, p + 44, 8);
            memcpy(&v1_blob_size, p + 52, 8);
            memcpy(&v1_file_size, p + 60, 8);
            memcpy(&v1_main_entry, p + 68, 8);
            memcpy(&v1_syscalls_ptr, p + 76, 8);
            
            /* Map to v2 struct layout */
            hdr->file_size = v1_file_size;
            hdr->nodes_offset = v1_nodes_offset;
            hdr->node_count = v1_node_count;
            hdr->edges_offset = v1_edges_offset;
            hdr->edge_count = v1_edge_count;
            hdr->blob_offset = v1_blob_offset;
            hdr->blob_size = v1_blob_size;
            hdr->cold_data_offset = 0;  /* v1 has no cold_data */
            hdr->cold_data_size = 0;
            hdr->main_entry_offset = v1_main_entry;
            hdr->syscalls_ptr_offset = v1_syscalls_ptr;
        }
        
        g->fd = fd;
        g->map_base = map;
        g->map_size = st.st_size;
        g->hdr = hdr;
        g->nodes = (Node *)((char *)map + hdr->nodes_offset);
        g->edges = (Edge *)((char *)map + hdr->edges_offset);
        g->blob = (uint8_t *)((char *)map + hdr->blob_offset);
        g->cold_data = (hdr->cold_data_size > 0) ? 
                       (uint8_t *)((char *)map + hdr->cold_data_offset) : NULL;
        g->node_count = hdr->node_count;
        g->edge_count = hdr->edge_count;
        g->blob_size = hdr->blob_size;
        g->cold_data_size = hdr->cold_data_size;
    }
    
    return g;
}

/* ========================================================================
 * SYNC TO DISK
 * ======================================================================== */

void melvin_sync(Graph *g) {
    if (!g || !g->map_base) return;
    msync(g->map_base, g->map_size, MS_SYNC);
}

/* ========================================================================
 * CLOSE
 * ======================================================================== */

void melvin_close(Graph *g) {
    if (!g) return;
    
    if (g->map_base) {
        msync(g->map_base, g->map_size, MS_SYNC);
        munmap(g->map_base, g->map_size);
    }
    
    if (g->fd >= 0) {
        close(g->fd);
    }
    
    free(g);
}

/* ========================================================================
 * SET SYSCALLS (writes pointer into blob)
 * ======================================================================== */

void melvin_set_syscalls(Graph *g, MelvinSyscalls *syscalls) {
    if (!g || !g->hdr || !syscalls) return;
    
    /* Write syscalls pointer into blob at known offset */
    if (g->hdr->syscalls_ptr_offset > 0 && 
        g->hdr->syscalls_ptr_offset < g->hdr->blob_size) {
        void **ptr_loc = (void **)(g->blob + g->hdr->syscalls_ptr_offset);
        *ptr_loc = syscalls;
    }
}

/* ========================================================================
 * GET SYSCALLS FROM BLOB (for blob code to call)
 * ======================================================================== */

MelvinSyscalls* melvin_get_syscalls_from_blob(Graph *g) {
    if (!g || !g->hdr) return NULL;
    
    if (g->hdr->syscalls_ptr_offset > 0 && 
        g->hdr->syscalls_ptr_offset < g->hdr->blob_size) {
        void **ptr_loc = (void **)(g->blob + g->hdr->syscalls_ptr_offset);
        return (MelvinSyscalls *)*ptr_loc;
    }
    
    return NULL;
}

/* ========================================================================
 * FEED BYTE (ONLY writes to .m, NO physics)
 * ======================================================================== */

static uint32_t find_edge(Graph *g, uint32_t src, uint32_t dst) {
    if (src >= g->node_count) return UINT32_MAX;  /* Hot region only */
    uint32_t eid = g->nodes[src].first_out;
    uint32_t max_iterations = (uint32_t)(g->edge_count + 1);  /* Safety: prevent infinite loops */
    uint32_t iterations = 0;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iterations < max_iterations) {
        if (g->edges[eid].dst == dst) return eid;
        eid = g->edges[eid].next_out;
        iterations++;
    }
    return UINT32_MAX;
}

static uint32_t create_edge(Graph *g, uint32_t src, uint32_t dst, float w) {
    /* Simple edge creation - just structure, no physics */
    /* Hot region only - edges stay in hot space */
    if (g->edge_count >= UINT32_MAX) {
        return UINT32_MAX;  /* Would need to grow file */
    }
    
    uint32_t eid = (uint32_t)g->edge_count++;
    g->hdr->edge_count = g->edge_count;  /* Sync to header */
    Edge *e = &g->edges[eid];
    e->src = src;
    e->dst = dst;
    e->w = w;
    e->next_out = g->nodes[src].first_out;
    e->next_in = g->nodes[dst].first_in;
    g->nodes[src].first_out = eid;
    g->nodes[dst].first_in = eid;
    g->nodes[src].out_degree++;
    g->nodes[dst].in_degree++;
    
    return eid;
}

void melvin_feed_byte(Graph *g, uint32_t port_node_id, uint8_t b, float energy) {
    if (!g || !g->hdr || port_node_id >= g->node_count) return;  /* Hot region only */
    
    uint32_t data_id = (uint32_t)b;
    if (data_id >= g->node_count) return;  /* Hot region only */
    
    /* ONLY write to mapped .m file - NO physics, NO stepping */
    g->nodes[port_node_id].a += energy;
    g->nodes[data_id].a += energy;
    
    /* Ensure edge exists (structure only) */
    if (find_edge(g, port_node_id, data_id) == UINT32_MAX) {
        create_edge(g, port_node_id, data_id, 0.1f);
    }
}

/* ========================================================================
 * UEL PHYSICS (embedded directly in melvin.c)
 * ======================================================================== */

/* UEL physics parameters */
static const struct {
    float eta_a;
    float eta_w;
    float lambda;
    float decay_a;
    float decay_w;
} uel_params = {
    .eta_a = 0.1f,
    .eta_w = 0.01f,
    .lambda = 0.05f,
    .decay_a = 0.05f,
    .decay_w = 0.001f
};

/* Helper: Find edge (hot region only) */
static uint32_t uel_find_edge(Graph *g, uint32_t src, uint32_t dst) {
    if (src >= g->node_count) return UINT32_MAX;  /* Hot region only */
    uint32_t eid = g->nodes[src].first_out;
    uint32_t max_iterations = (uint32_t)(g->edge_count + 1);  /* Safety: prevent infinite loops */
    uint32_t iterations = 0;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iterations < max_iterations) {
        if (g->edges[eid].dst == dst) return eid;
        eid = g->edges[eid].next_out;
        iterations++;
    }
    return UINT32_MAX;
}

/* Kernel function K(i,j) - hot region only */
static inline float uel_kernel(Graph *g, uint32_t i, uint32_t j) {
    if (i == j || i >= g->node_count || j >= g->node_count) return 0.0f;  /* Hot region only */
    
    if (uel_find_edge(g, j, i) != UINT32_MAX || uel_find_edge(g, i, j) != UINT32_MAX) {
        return 0.5f;
    }
    
    /* Shared neighbor check - hot region only */
    uint32_t ei = g->nodes[i].first_in;
    uint32_t max_iter_i = (uint32_t)(g->edge_count + 1);
    uint32_t iter_i = 0;
    
    while (ei != UINT32_MAX && ei < g->edge_count && iter_i < max_iter_i) {
        uint32_t neighbor = g->edges[ei].src;
        uint32_t ej = g->nodes[j].first_in;
        uint32_t max_iter_j = (uint32_t)(g->edge_count + 1);
        uint32_t iter_j = 0;
        
        while (ej != UINT32_MAX && ej < g->edge_count && iter_j < max_iter_j) {
            if (g->edges[ej].src == neighbor) return 0.3f;
            ej = g->edges[ej].next_in;
            iter_j++;
        }
        ei = g->edges[ei].next_in;
        iter_i++;
    }
    
    return 0.01f;
}

/* Compute mass - hot region only */
static inline float uel_compute_mass(Graph *g, uint32_t i) {
    if (i >= g->node_count) return 0.0f;  /* Hot region only */
    float a_abs = fabsf(g->nodes[i].a);
    float degree = (float)(g->nodes[i].in_degree + g->nodes[i].out_degree);
    return a_abs + 0.1f * degree;
}

/* UEL TICK FUNCTION - Universal Emergence Law (hot region only) */
static void uel_main(Graph *g) {
    if (!g || !g->hdr || g->node_count == 0) return;
    
    uint64_t N = g->node_count;  /* Hot region only - never touches cold_data */
    
    /* Allocate scratch buffers (these are temporary, not in .m) */
    float *msg = calloc(N, sizeof(float));
    float *phi = calloc(N, sizeof(float));
    float *mass = calloc(N, sizeof(float));
    
    if (!msg || !phi || !mass) {
        free(msg); free(phi); free(mass);
        return;
    }
    
    /* PHASE 1: Compute mass */
    for (size_t i = 0; i < N; i++) {
        mass[i] = uel_compute_mass(g, (uint32_t)i);
    }
    
    /* PHASE 2: Compute global field Φ */
    memset(phi, 0, N * sizeof(float));
    
    uint32_t active[1024];
    size_t num_active = 0;
    for (size_t j = 0; j < N && num_active < 1024; j++) {
        if (mass[j] > 0.01f) {
            active[num_active++] = (uint32_t)j;
        }
    }
    
    for (uint64_t i = 0; i < N; i++) {
        if (mass[i] < 0.001f) continue;
        float phi_i = 0.0f;
        for (size_t k = 0; k < num_active; k++) {
            uint32_t j = active[k];
            if (j != (uint32_t)i) phi_i += mass[j] * uel_kernel(g, (uint32_t)i, j);
        }
        phi[i] = phi_i;
    }
    
    /* PHASE 3: Local messages - hot region only */
    memset(msg, 0, (size_t)(N * sizeof(float)));
    for (uint64_t i = 0; i < N; i++) {
        uint32_t eid = g->nodes[i].first_in;
        uint32_t max_iterations = (uint32_t)(g->edge_count + 1);  /* Safety: prevent infinite loops */
        uint32_t iterations = 0;
        
        while (eid != UINT32_MAX && eid < g->edge_count && iterations < max_iterations) {
            msg[i] += g->edges[eid].w * g->nodes[g->edges[eid].src].a;
            eid = g->edges[eid].next_in;
            iterations++;
        }
    }
    
    /* PHASE 4: Update activations (minimize chaos via gradient descent on F) - hot region only */
    /* UEL: da_i/dt = -η_a * ∂F/∂a_i */
    /* ∂F/∂a_i includes: chaos term (incoherence with neighbors) + activation cost */
    for (uint64_t i = 0; i < N; i++) {
        float a_i = g->nodes[i].a;
        float msg_i = msg[i];
        float phi_i = phi[i];
        
        /* Combined input from neighbors and global field */
        float field_input = msg_i + uel_params.lambda * phi_i;
        
        /* Local chaos (incoherence): how much a_i disagrees with neighbors */
        float chaos_i = (a_i - msg_i) * (a_i - msg_i);
        
        /* Gradient descent: move a_i toward field_input to reduce chaos */
        /* This is NOT "prediction" - it's just energy minimization */
        float da_i = -uel_params.eta_a * (a_i - field_input);
        
        /* Update with decay (activation cost term) */
        float new_a = a_i + da_i - uel_params.decay_a * a_i;
        g->nodes[i].a = tanhf(new_a);
    }
    
    /* PHASE 5: Update weights (minimize chaos via gradient descent on F) - hot region only */
    /* UEL: dW_ij/dt = -η_w * ∂F/∂W_ij */
    /* ∂F/∂W_ij includes: chaos term (edges causing incoherence are penalized) */
    for (uint64_t eid = 0; eid < g->edge_count; eid++) {  /* Hot region only */
        uint32_t dst = g->edges[eid].dst;
        uint32_t src = g->edges[eid].src;
        if (dst >= N || src >= N) continue;
        
        float a_dst = g->nodes[dst].a;
        float a_src = g->nodes[src].a;
        float msg_dst = msg[dst];
        
        /* Local chaos at destination: incoherence between activation and neighbors */
        /* This is NOT "prediction error" - it's just measuring chaos */
        float chaos_dst = a_dst - msg_dst;
        
        /* Gradient: if edge causes chaos, reduce weight; if it reduces chaos, increase */
        /* Hebbian component: strengthen if source active when destination needs it */
        float dw = -uel_params.eta_w * chaos_dst * a_src;
        g->edges[eid].w += dw;
        
        /* Weight decay (structural inefficiency term C_i) */
        g->edges[eid].w *= (1.0f - uel_params.decay_w);
        
        /* Clamp to prevent explosion */
        if (g->edges[eid].w > 5.0f) g->edges[eid].w = 5.0f;
        if (g->edges[eid].w < -5.0f) g->edges[eid].w = -5.0f;
    }
    
    free(msg);
    free(phi);
    free(mass);
}

/* ========================================================================
 * CREATE V2 FILE (reusable for corpus packing)
 * ======================================================================== */

int melvin_create_v2(const char *path, 
                     uint64_t hot_nodes, 
                     uint64_t hot_edges, 
                     uint64_t hot_blob_bytes, 
                     uint64_t cold_data_bytes) {
    if (!path) return -1;
    
    /* Calculate layout - all offsets 64-bit */
    uint64_t header_size = sizeof(MelvinHeader);
    uint64_t nodes_size = hot_nodes * sizeof(Node);
    uint64_t edges_size = hot_edges * sizeof(Edge);
    
    uint64_t off = header_size;
    uint64_t nodes_offset = off;
    off += nodes_size;
    uint64_t edges_offset = off;
    off += edges_size;
    uint64_t blob_offset = off;
    off += hot_blob_bytes;
    uint64_t cold_data_offset = off;
    off += cold_data_bytes;
    uint64_t total_size = off;
    
    /* Round up to page size */
    size_t page_size = getpagesize();
    total_size = (total_size + page_size - 1) & ~(page_size - 1);
    
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return -1;
    }
    
    /* Create file */
    if (ftruncate(fd, total_size) < 0) {
        close(fd);
        return -1;
    }
    
    /* mmap */
    void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return -1;
    }
    
    /* Initialize header */
    MelvinHeader *hdr = (MelvinHeader *)map;
    memcpy(hdr->magic, MELVIN_MAGIC, 4);
    hdr->version = MELVIN_VERSION;
    hdr->flags = 0;
    hdr->file_size = total_size;
    hdr->nodes_offset = nodes_offset;
    hdr->node_count = hot_nodes;
    hdr->edges_offset = edges_offset;
    hdr->edge_count = 0;
    hdr->blob_offset = blob_offset;
    hdr->blob_size = hot_blob_bytes;
    hdr->cold_data_offset = cold_data_offset;
    hdr->cold_data_size = cold_data_bytes;
    hdr->main_entry_offset = 0;  /* Set by tool that seeds blob */
    hdr->syscalls_ptr_offset = 0; /* Set by tool */
    
    /* Zero hot regions */
    memset((char *)map + hdr->nodes_offset, 0, (size_t)nodes_size);
    memset((char *)map + hdr->edges_offset, 0, (size_t)edges_size);
    memset((char *)map + hdr->blob_offset, 0, (size_t)hot_blob_bytes);
    /* Cold data left as-is (will be filled by corpus loader) */
    
    /* Initialize data nodes (0-255) - just structure, no physics */
    Node *nodes = (Node *)((char *)map + hdr->nodes_offset);
    for (int i = 0; i < 256 && i < (int)hot_nodes; i++) {
        nodes[i].byte = (uint8_t)i;
        nodes[i].a = 0.0f;
        nodes[i].first_in = UINT32_MAX;
        nodes[i].first_out = UINT32_MAX;
        nodes[i].in_degree = 0;
        nodes[i].out_degree = 0;
    }
    
    /* Sync and cleanup */
    msync(map, total_size, MS_SYNC);
    munmap(map, total_size);
    close(fd);
    
    return 0;
}

/* ========================================================================
 * CALL ENTRY (run UEL physics)
 * ======================================================================== */

void melvin_call_entry(Graph *g) {
    if (!g || !g->hdr) return;
    
    /* Run UEL physics (embedded in melvin.c) */
    uel_main(g);
}

/* ========================================================================
 * DEBUG HELPERS (read-only inspection)
 * ======================================================================== */

float melvin_get_activation(Graph *g, uint32_t node_id) {
    if (!g || !g->hdr || node_id >= g->node_count) return 0.0f;  /* Hot region only */
    return g->nodes[node_id].a;
}

/* ========================================================================
 * COLD DATA ACCESS (graph can copy from cold to hot)
 * ======================================================================== */

/* Copy bytes from cold_data to hot blob - graph-accessible via machine code */
void melvin_copy_from_cold(Graph *g, uint64_t cold_offset, uint64_t length, uint64_t blob_target_offset) {
    if (!g || !g->cold_data || !g->blob) return;
    
    /* Validate cold access */
    if (cold_offset >= g->cold_data_size) return;
    if (cold_offset + length > g->cold_data_size) {
        length = g->cold_data_size - cold_offset;  /* Clamp to available */
    }
    
    /* Validate blob target */
    if (blob_target_offset >= g->blob_size) return;
    if (blob_target_offset + length > g->blob_size) {
        length = g->blob_size - blob_target_offset;  /* Clamp to available */
    }
    
    /* Copy from cold to hot */
    memcpy(g->blob + blob_target_offset, g->cold_data + cold_offset, (size_t)length);
}

