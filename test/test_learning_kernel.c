#define _POSIX_C_SOURCE 200809L

/*
 * test_learning_kernel.c
 * 
 * Minimal test to prove that strengthen_edges_with_prediction_and_reward()
 * can actually change edge weights in isolation.
 * 
 * This bypasses everything else (file I/O, complex graph, etc.) to test
 * the learning kernel directly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "melvin.c"

// ========================================================================
// MINIMAL GRAPH SETUP
// ========================================================================

// File must stay in scope, so we return both
typedef struct {
    MelvinRuntime *rt;
    MelvinFile file;
} RuntimeAndFile;

static RuntimeAndFile* create_minimal_runtime(void) {
    // Create a minimal .m file in memory
    const char *test_file = "test_kernel.m";
    
    // Remove old file if it exists
    unlink(test_file);
    
    // Initialize new file
    GraphParams params = {0};
    params.decay_rate = 0.97f;
    params.learning_rate = 0.1f;  // High learning rate for kernel test
    params.homeostasis_target = 0.5f;
    params.exec_threshold = 1.0f;
    
    printf("[create_minimal_runtime] Creating test file...\n");
    if (melvin_m_init_new_file(test_file, &params) < 0) {
        fprintf(stderr, "Failed to create test file\n");
        return NULL;
    }
    
    printf("[create_minimal_runtime] Mapping test file...\n");
    
    // Allocate structure to hold both runtime and file
    RuntimeAndFile *rf = calloc(1, sizeof(RuntimeAndFile));
    if (!rf) {
        fprintf(stderr, "Failed to allocate RuntimeAndFile\n");
        return NULL;
    }
    
    if (melvin_m_map(test_file, &rf->file) < 0) {
        fprintf(stderr, "Failed to map test file\n");
        free(rf);
        return NULL;
    }
    
    printf("[create_minimal_runtime] Allocating runtime...\n");
    // Allocate runtime
    rf->rt = calloc(1, sizeof(MelvinRuntime));
    if (!rf->rt) {
        fprintf(stderr, "Failed to allocate runtime\n");
        close_file(&rf->file);
        free(rf);
        return NULL;
    }
    
    printf("[create_minimal_runtime] Initializing runtime...\n");
    fflush(stdout);
    int init_result = runtime_init(rf->rt, &rf->file);
    if (init_result < 0) {
        fprintf(stderr, "Failed to init runtime (result=%d)\n", init_result);
        free(rf->rt);
        close_file(&rf->file);
        free(rf);
        return NULL;
    }
    
    printf("[create_minimal_runtime] Runtime initialized successfully.\n");
    fflush(stdout);
    return rf;
}

static void setup_minimal_graph(MelvinRuntime *rt) {
    if (!rt) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: rt is NULL\n");
        return;
    }
    
    printf("[setup_minimal_graph] rt=%p rt->file=%p\n", (void*)rt, (void*)rt->file);
    fflush(stdout);
    
    if (!rt->file) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: rt->file is NULL\n");
        return;
    }
    
    // Refresh all pointers from file structure
    MelvinFile *file = rt->file;
    printf("[setup_minimal_graph] file=%p\n", (void*)file);
    printf("[setup_minimal_graph] file->map=%p file->file_header=%p\n",
           (void*)file->map, (void*)file->file_header);
    fflush(stdout);
    
    if (!file->map || !file->file_header) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: file->map or file->file_header is NULL\n");
        return;
    }
    
    printf("[setup_minimal_graph] file->graph_header=%p (before check)\n", 
           (void*)file->graph_header);
    fflush(stdout);
    
    // Recalculate graph_header from map (in case pointer is stale)
    if (file->file_header->graph_header_offset > 0) {
        GraphHeaderDisk *gh_recalc = (GraphHeaderDisk*)((uint8_t*)file->map + file->file_header->graph_header_offset);
        printf("[setup_minimal_graph] Recalculated graph_header=%p (offset=%llu)\n",
               (void*)gh_recalc, (unsigned long long)file->file_header->graph_header_offset);
        file->graph_header = gh_recalc;
    }
    
    GraphHeaderDisk *gh = file->graph_header;
    
    if (!gh) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: graph_header is NULL\n");
        return;
    }
    
    // Check if pointer looks valid (on ARM64, high bits can be set)
    // Just check it's not NULL and not obviously wrong (too small)
    if ((uintptr_t)gh < 0x1000) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: graph_header pointer looks invalid: %p\n", (void*)gh);
        return;
    }
    
    printf("[setup_minimal_graph] gh=%p, reading values...\n", (void*)gh);
    fflush(stdout);
    
    // Read values carefully
    uint64_t num_nodes = gh->num_nodes;
    uint64_t node_capacity = gh->node_capacity;
    uint64_t num_edges = gh->num_edges;
    uint64_t edge_capacity = gh->edge_capacity;
    
    printf("[setup_minimal_graph] Raw values: nodes=%llu capacity=%llu edges=%llu e_capacity=%llu\n",
           (unsigned long long)num_nodes, (unsigned long long)node_capacity,
           (unsigned long long)num_edges, (unsigned long long)edge_capacity);
    fflush(stdout);
    
    // Check if values look reasonable
    if (num_nodes > 10000 || node_capacity > 100000 || num_edges > 100000 || edge_capacity > 100000) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: Values look like garbage (corrupted memory)\n");
        return;
    }
    
    if (!file->nodes || !file->edges) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: Invalid node/edge arrays\n");
        return;
    }
    
    NodeDisk *nodes = file->nodes;
    EdgeDisk *edges = file->edges;
    
    // Create 2 nodes: src and dst
    uint64_t src_id = 1000ULL;
    uint64_t dst_id = 2000ULL;
    
    // Check if we have enough capacity (don't try to grow - file might not support it)
    if (gh->num_nodes + 2 > gh->node_capacity) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: Not enough node capacity (%llu/%llu)\n",
                (unsigned long long)gh->num_nodes, (unsigned long long)gh->node_capacity);
        return;
    }
    
    // Ensure edge capacity too
    if (gh->num_edges + 1 > gh->edge_capacity) {
        fprintf(stderr, "[setup_minimal_graph] ERROR: Not enough edge capacity (%llu/%llu)\n",
                (unsigned long long)gh->num_edges, (unsigned long long)gh->edge_capacity);
        return;
    }
    
    // Find or create src node (use file pointer directly)
    uint64_t src_idx = find_node_index_by_id(file, src_id);
    if (src_idx == UINT64_MAX) {
        src_idx = gh->num_nodes++;
        memset(&nodes[src_idx], 0, sizeof(NodeDisk));  // Zero all fields
        nodes[src_idx].id = src_id;
        nodes[src_idx].state = 1.0f;  // High activation
        nodes[src_idx].flags = 0;
        nodes[src_idx].prediction_error = 0.0f;
        nodes[src_idx].reward = 0.0f;
        nodes[src_idx].first_out_edge = UINT64_MAX;
        nodes[src_idx].out_degree = 0;
    }
    
    // Find or create dst node
    uint64_t dst_idx = find_node_index_by_id(file, dst_id);
    if (dst_idx == UINT64_MAX) {
        dst_idx = gh->num_nodes++;
        memset(&nodes[dst_idx], 0, sizeof(NodeDisk));  // Zero all fields
        nodes[dst_idx].id = dst_id;
        nodes[dst_idx].state = 0.5f;  // Some activation
        nodes[dst_idx].flags = 0;
        nodes[dst_idx].prediction_error = 1.0f;  // NONZERO prediction error!
        nodes[dst_idx].reward = 0.0f;
        nodes[dst_idx].first_out_edge = UINT64_MAX;
        nodes[dst_idx].out_degree = 0;
    }
    
    // Create edge: src -> dst using create_edge_between helper
    if (!edge_exists_between(file, src_id, dst_id)) {
        create_edge_between(file, src_id, dst_id, 0.2f);
    }
    
    // Refresh pointers after potential realloc in create_edge_between
    gh = file->graph_header;
    edges = file->edges;
    nodes = file->nodes;
    
    // Find the edge we just created
    uint64_t edge_idx = UINT64_MAX;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (edges[i].src == src_id && edges[i].dst == dst_id) {
            edge_idx = i;
            break;
        }
    }
    
    if (edge_idx != UINT64_MAX) {
        // Set up high trace and eligibility for learning
        edges[edge_idx].trace = 100.0f;  // High trace (edge is used)
        edges[edge_idx].eligibility = 1.0f;  // High eligibility (ready to learn)
        edges[edge_idx].usage = 0.5f;
        edges[edge_idx].age = 100;
    }
    
    // Set learning rate high for this test
    gh->learning_rate = 0.1f;
    gh->reward_lambda = 1.0f;
    
    printf("[kernel_setup] Created minimal graph:\n");
    printf("  src node: id=%llu, state=%.3f\n", (unsigned long long)src_id, nodes[src_idx].state);
    printf("  dst node: id=%llu, state=%.3f, prediction_error=%.3f\n", 
           (unsigned long long)dst_id, nodes[dst_idx].state, nodes[dst_idx].prediction_error);
    if (edge_idx != UINT64_MAX) {
        printf("  edge: src->dst, weight=%.6f, trace=%.3f, eligibility=%.6f\n",
               edges[edge_idx].weight, edges[edge_idx].trace, edges[edge_idx].eligibility);
    }
    printf("  learning_rate=%.6f\n", gh->learning_rate);
}

static float get_edge_weight(MelvinRuntime *rt, uint64_t src_id, uint64_t dst_id) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    EdgeDisk *edges = rt->file->edges;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (edges[i].src == src_id && edges[i].dst == dst_id) {
            return edges[i].weight;
        }
    }
    return 0.0f;
}

// ========================================================================
// KERNEL TEST
// ========================================================================

static void test_learning_kernel(void) {
    printf("\n========================================\n");
    printf("LEARNING KERNEL TEST\n");
    printf("========================================\n\n");
    fflush(stdout);
    
    printf("[test_learning_kernel] Creating runtime...\n");
    fflush(stdout);
    RuntimeAndFile *rf = create_minimal_runtime();
    if (!rf || !rf->rt) {
        printf("FAILED: Could not create runtime\n");
        fflush(stdout);
        return;
    }
    
    MelvinRuntime *rt = rf->rt;
    
    // Verify file structure is valid
    if (!rt->file || !rt->file->graph_header) {
        printf("FAILED: File structure invalid after runtime_init\n");
        fflush(stdout);
        runtime_cleanup(rt);
        close_file(&rf->file);
        free(rf->rt);
        free(rf);
        return;
    }
    
    GraphHeaderDisk *gh_check = rt->file->graph_header;
    printf("[test_learning_kernel] File check: nodes=%llu/%llu edges=%llu/%llu\n",
           (unsigned long long)gh_check->num_nodes,
           (unsigned long long)gh_check->node_capacity,
           (unsigned long long)gh_check->num_edges,
           (unsigned long long)gh_check->edge_capacity);
    fflush(stdout);
    
    printf("[test_learning_kernel] Setting up minimal graph...\n");
    fflush(stdout);
    setup_minimal_graph(rt);
    printf("[test_learning_kernel] Graph setup complete.\n");
    fflush(stdout);
    
    uint64_t src_id = 1000ULL;
    uint64_t dst_id = 2000ULL;
    
    float w0 = get_edge_weight(rt, src_id, dst_id);
    printf("\n[KERNEL_TEST] Initial weight: %.6f\n", w0);
    
    if (w0 == 0.0f) {
        printf("FAILED: Edge not found!\n");
        runtime_cleanup(rt);
        close_file(rt->file);
        free(rt);
        return;
    }
    
    // Call learning kernel in a loop
    printf("\n[KERNEL_TEST] Calling strengthen_edges_with_prediction_and_reward() 1000 times...\n");
    
    for (int i = 0; i < 1000; i++) {
        // Before each call, refresh prediction_error (it might decay)
        NodeDisk *nodes = rt->file->nodes;
        GraphHeaderDisk *gh = rt->file->graph_header;
        
        uint64_t dst_idx = find_node_index_by_id(rt->file, dst_id);
        if (dst_idx != UINT64_MAX) {
            // Keep prediction_error nonzero
            nodes[dst_idx].prediction_error = 1.0f;
        }
        
        // Call the learning kernel
        strengthen_edges_with_prediction_and_reward(rt);
        
        // Print progress every 100 iterations
        if ((i + 1) % 100 == 0) {
            float w_current = get_edge_weight(rt, src_id, dst_id);
            printf("[KERNEL_TEST] Iteration %d: weight=%.6f (change: %.6f)\n",
                   i + 1, w_current, w_current - w0);
        }
    }
    
    float w_final = get_edge_weight(rt, src_id, dst_id);
    float delta = w_final - w0;
    
    printf("\n[KERNEL_TEST] Final weight: %.6f\n", w_final);
    printf("[KERNEL_TEST] Total change: %.6f\n", delta);
    printf("[KERNEL_TEST] Change per iteration: %.9f\n", delta / 1000.0f);
    
    // ASSERTION: Weight must have increased
    if (w_final > w0 + 0.0001f) {
        printf("\n[KERNEL_TEST] ✓ PASS: Weight increased from %.6f to %.6f\n", w0, w_final);
        printf("[KERNEL_TEST] The learning kernel CAN change weights!\n");
    } else {
        printf("\n[KERNEL_TEST] ✗ FAIL: Weight did not increase significantly\n");
        printf("[KERNEL_TEST] Initial: %.6f, Final: %.6f, Change: %.9f\n", w0, w_final, delta);
        printf("[KERNEL_TEST] The learning kernel is NOT working!\n");
        printf("\n[KERNEL_TEST] Possible causes:\n");
        printf("  1. Learning formula is mathematically zero\n");
        printf("  2. Eligibility decay is too fast\n");
        printf("  3. Weight saturation is clamping too early\n");
        printf("  4. Prediction error is being zeroed\n");
    }
    
    // Cleanup
    runtime_cleanup(rt);
    close_file(&rf->file);
    free(rf->rt);
    free(rf);
    
    // Remove test file
    unlink("test_kernel.m");
}

// ========================================================================
// MAIN
// ========================================================================

int main(int argc, char **argv) {
    test_learning_kernel();
    return 0;
}

