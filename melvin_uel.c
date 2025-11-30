/*
 * melvin_uel.c - UEL Physics Implementation
 * 
 * This file contains the Universal Emergence Law physics.
 * 
 * IMPORTANT: This is NOT linked into the runtime loader.
 * 
 * It is used in TWO ways:
 *   1. As source for a tool that extracts machine code into .m blob
 *   2. As a reference implementation
 * 
 * The runtime (melvin.c) NEVER calls functions from this file.
 * All physics runs from machine code inside the .m blob.
 */

#include "melvin.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* Helper: Find edge */
static uint32_t uel_find_edge(Graph *g, uint32_t src, uint32_t dst) {
    if (src >= g->hdr->node_count) return UINT32_MAX;
    uint32_t eid = g->nodes[src].first_out;
    uint32_t max_iterations = g->hdr->edge_count + 1;  /* Safety: prevent infinite loops */
    uint32_t iterations = 0;
    
    while (eid != UINT32_MAX && eid < g->hdr->edge_count && iterations < max_iterations) {
        if (g->edges[eid].dst == dst) return eid;
        eid = g->edges[eid].next_out;
        iterations++;
    }
    return UINT32_MAX;
}

/* Kernel function K(i,j) */
static inline float uel_kernel(Graph *g, uint32_t i, uint32_t j) {
    if (i == j || i >= g->hdr->node_count || j >= g->hdr->node_count) return 0.0f;
    
    if (uel_find_edge(g, j, i) != UINT32_MAX || uel_find_edge(g, i, j) != UINT32_MAX) {
        return 0.5f;
    }
    
    /* Shared neighbor check */
    uint32_t ei = g->nodes[i].first_in;
    uint32_t max_iter_i = g->hdr->edge_count + 1;
    uint32_t iter_i = 0;
    
    while (ei != UINT32_MAX && ei < g->hdr->edge_count && iter_i < max_iter_i) {
        uint32_t neighbor = g->edges[ei].src;
        uint32_t ej = g->nodes[j].first_in;
        uint32_t max_iter_j = g->hdr->edge_count + 1;
        uint32_t iter_j = 0;
        
        while (ej != UINT32_MAX && ej < g->hdr->edge_count && iter_j < max_iter_j) {
            if (g->edges[ej].src == neighbor) return 0.3f;
            ej = g->edges[ej].next_in;
            iter_j++;
        }
        ei = g->edges[ei].next_in;
        iter_i++;
    }
    
    return 0.01f;
}

/* Compute mass */
static inline float uel_compute_mass(Graph *g, uint32_t i) {
    if (i >= g->hdr->node_count) return 0.0f;
    float a_abs = fabsf(g->nodes[i].a);
    float degree = (float)(g->nodes[i].in_degree + g->nodes[i].out_degree);
    return a_abs + 0.1f * degree;
}

/* 
 * UEL TICK FUNCTION
 * 
 * This function implements the Universal Emergence Law.
 * 
 * When compiled, its machine code should be extracted and written
 * into the .m blob. The runtime calls this via melvin_call_entry().
 * 
 * Signature: void uel_main(Graph *g)
 * 
 * This function:
 *   - Computes global field phi
 *   - Computes local messages
 *   - Updates activations
 *   - Updates weights
 *   - Handles all physics loops
 * 
 * The runtime (melvin.c) does NONE of this - it only jumps here.
 */
void uel_main(Graph *g) {
    if (!g || !g->hdr || g->hdr->node_count == 0) return;
    
    size_t N = g->hdr->node_count;
    
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
    
    for (size_t i = 0; i < N; i++) {
        if (mass[i] < 0.001f) continue;
        float phi_i = 0.0f;
        for (size_t k = 0; k < num_active; k++) {
            uint32_t j = active[k];
            if (j != i) phi_i += mass[j] * uel_kernel(g, (uint32_t)i, j);
        }
        phi[i] = phi_i;
    }
    
    /* PHASE 3: Local messages */
    memset(msg, 0, N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        uint32_t eid = g->nodes[i].first_in;
        uint32_t max_iterations = g->hdr->edge_count + 1;  /* Safety: prevent infinite loops */
        uint32_t iterations = 0;
        
        while (eid != UINT32_MAX && eid < g->hdr->edge_count && iterations < max_iterations) {
            msg[i] += g->edges[eid].w * g->nodes[g->edges[eid].src].a;
            eid = g->edges[eid].next_in;
            iterations++;
        }
    }
    
    /* PHASE 4: Update activations (minimize chaos via gradient descent on F) */
    /* UEL: da_i/dt = -η_a * ∂F/∂a_i */
    /* ∂F/∂a_i includes: chaos term (incoherence with neighbors) + activation cost */
    for (size_t i = 0; i < N; i++) {
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
    
    /* PHASE 5: Update weights (minimize chaos via gradient descent on F) */
    /* UEL: dW_ij/dt = -η_w * ∂F/∂W_ij */
    /* ∂F/∂W_ij includes: chaos term (edges causing incoherence are penalized) */
    for (size_t eid = 0; eid < g->hdr->edge_count; eid++) {
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
