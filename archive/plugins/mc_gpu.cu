// GPU Acceleration Plugin for Melvin
// Provides CUDA-accelerated graph operations
// Compile with: nvcc -shared -Xcompiler -fPIC -arch=sm_87 -o mc_gpu.so mc_gpu.cu -lcudart

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

// Export C symbols (no name mangling)
extern "C" {

// Melvin graph structures (from melvin.h)
typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t tick;
    uint64_t node_cap;
    uint64_t edge_cap;
} BrainHeader;

typedef struct {
    float a;
    float bias;
    float decay;
    uint32_t kind;
    uint32_t flags;
    float reliability;
    uint32_t success_count;
    uint32_t failure_count;
    uint32_t mc_id;
    uint16_t mc_flags;
    uint16_t mc_role;
    float value;
} Node;

typedef struct {
    uint64_t src;
    uint64_t dst;
    float w;
    uint32_t flags;
    float elig;
    uint32_t usage_count;
} Edge;

typedef struct {
    BrainHeader *header;
    Node *nodes;
    Edge *edges;
    size_t mmap_size;
    int fd;
} Brain;

// GPU state
static float *d_nodes_a = NULL;      // Device copy of node activations
static float *d_nodes_bias = NULL;   // Device copy of node biases
static float *d_predicted_a = NULL;  // Device copy of predicted activations
static float *d_node_error = NULL;   // Device copy of node errors
static float *d_edges_w = NULL;      // Device copy of edge weights
static uint64_t *d_edges_src = NULL; // Device copy of edge sources
static uint64_t *d_edges_dst = NULL; // Device copy of edge destinations
static uint64_t gpu_node_cap = 0;
static uint64_t gpu_edge_cap = 0;
static int gpu_initialized = 0;

// CUDA kernel: Propagate predictions
__global__ void propagate_kernel(
    float *nodes_a,
    float *nodes_bias,
    float *predicted_a,
    uint64_t *edges_src,
    uint64_t *edges_dst,
    float *edges_w,
    uint64_t num_nodes,
    uint64_t num_edges
) {
    uint64_t node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;
    
    // Start with bias
    float sum = nodes_bias[node_idx];
    
    // Sum weighted inputs from edges
    for (uint64_t i = 0; i < num_edges; i++) {
        if (edges_dst[i] == node_idx) {
            uint64_t src = edges_src[i];
            if (src < num_nodes) {
                sum += edges_w[i] * nodes_a[src];
            }
        }
    }
    
    // Apply sigmoid and clamp
    sum = 1.0f / (1.0f + expf(-sum));
    if (sum < 0.0f) sum = 0.0f;
    if (sum > 1.0f) sum = 1.0f;
    
    predicted_a[node_idx] = sum;
}

// CUDA kernel: Compute errors
__global__ void compute_error_kernel(
    float *nodes_a,
    float *predicted_a,
    float *node_error,
    uint64_t num_nodes
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    float err = nodes_a[idx] - predicted_a[idx];
    if (err > 1.0f) err = 1.0f;
    if (err < -1.0f) err = -1.0f;
    node_error[idx] = err;
}

// CUDA kernel: Update edge weights
__global__ void update_edges_kernel(
    float *nodes_a,
    float *node_error,
    uint64_t *edges_src,
    uint64_t *edges_dst,
    float *edges_w,
    uint64_t num_edges,
    uint64_t num_nodes,
    float eta
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    
    uint64_t src = edges_src[idx];
    uint64_t dst = edges_dst[idx];
    
    if (src >= num_nodes || dst >= num_nodes) return;
    
    float a_src = nodes_a[src];
    float err_dst = node_error[dst];
    
    // Learning rule: Δw = η * a_src * err_dst
    float dw = eta * a_src * err_dst;
    
    // NaN guard
    if (dw != dw || dw > 1e10f || dw < -1e10f) dw = 0.0f;
    
    edges_w[idx] += dw;
    
    // Clamp weights
    const float W_MAX = 10.0f;
    if (edges_w[idx] > W_MAX) edges_w[idx] = W_MAX;
    if (edges_w[idx] < -W_MAX) edges_w[idx] = -W_MAX;
}

// Initialize GPU memory
static int init_gpu_memory(Brain *g) {
    if (gpu_initialized && gpu_node_cap >= g->header->node_cap && 
        gpu_edge_cap >= g->header->edge_cap) {
        return 1; // Already initialized with enough capacity
    }
    
    uint64_t n = g->header->node_cap;
    uint64_t e = g->header->edge_cap;
    
    // Free old memory if resizing
    if (gpu_initialized) {
        if (d_nodes_a) cudaFree(d_nodes_a);
        if (d_nodes_bias) cudaFree(d_nodes_bias);
        if (d_predicted_a) cudaFree(d_predicted_a);
        if (d_node_error) cudaFree(d_node_error);
        if (d_edges_w) cudaFree(d_edges_w);
        if (d_edges_src) cudaFree(d_edges_src);
        if (d_edges_dst) cudaFree(d_edges_dst);
    }
    
    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc((void**)&d_nodes_a, n * sizeof(float));
    if (err != cudaSuccess) { printf("[mc_gpu] Failed to allocate d_nodes_a\n"); return 0; }
    
    err = cudaMalloc((void**)&d_nodes_bias, n * sizeof(float));
    if (err != cudaSuccess) { printf("[mc_gpu] Failed to allocate d_nodes_bias\n"); return 0; }
    
    err = cudaMalloc((void**)&d_predicted_a, n * sizeof(float));
    if (err != cudaSuccess) { printf("[mc_gpu] Failed to allocate d_predicted_a\n"); return 0; }
    
    err = cudaMalloc((void**)&d_node_error, n * sizeof(float));
    if (err != cudaSuccess) { printf("[mc_gpu] Failed to allocate d_node_error\n"); return 0; }
    
    err = cudaMalloc((void**)&d_edges_w, e * sizeof(float));
    if (err != cudaSuccess) { printf("[mc_gpu] Failed to allocate d_edges_w\n"); return 0; }
    
    err = cudaMalloc((void**)&d_edges_src, e * sizeof(uint64_t));
    if (err != cudaSuccess) { printf("[mc_gpu] Failed to allocate d_edges_src\n"); return 0; }
    
    err = cudaMalloc((void**)&d_edges_dst, e * sizeof(uint64_t));
    if (err != cudaSuccess) { printf("[mc_gpu] Failed to allocate d_edges_dst\n"); return 0; }
    
    gpu_node_cap = n;
    gpu_edge_cap = e;
    gpu_initialized = 1;
    
    printf("[mc_gpu] GPU memory initialized: %llu nodes, %llu edges\n",
           (unsigned long long)n, (unsigned long long)e);
    return 1;
}

// MC Function: GPU-accelerated propagation
void mc_gpu_propagate(Brain *g, uint64_t node_id) {
    if (!init_gpu_memory(g)) {
        printf("[mc_gpu] GPU initialization failed, skipping\n");
        return;
    }
    
    uint64_t n = g->header->num_nodes;
    uint64_t e = g->header->num_edges;
    
    // Copy node data to GPU
    float *h_nodes_a = (float*)malloc(n * sizeof(float));
    float *h_nodes_bias = (float*)malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) {
        h_nodes_a[i] = g->nodes[i].a;
        h_nodes_bias[i] = g->nodes[i].bias;
    }
    
    cudaMemcpy(d_nodes_a, h_nodes_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_bias, h_nodes_bias, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy edge data to GPU
    uint64_t *h_edges_src = (uint64_t*)malloc(e * sizeof(uint64_t));
    uint64_t *h_edges_dst = (uint64_t*)malloc(e * sizeof(uint64_t));
    float *h_edges_w = (float*)malloc(e * sizeof(float));
    for (uint64_t i = 0; i < e; i++) {
        h_edges_src[i] = g->edges[i].src;
        h_edges_dst[i] = g->edges[i].dst;
        h_edges_w[i] = g->edges[i].w;
    }
    
    cudaMemcpy(d_edges_src, h_edges_src, e * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges_dst, h_edges_dst, e * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges_w, h_edges_w, e * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    propagate_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_nodes_a, d_nodes_bias, d_predicted_a,
        d_edges_src, d_edges_dst, d_edges_w,
        n, e
    );
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[mc_gpu] Kernel error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy results back (if needed for error computation)
    // For now, we'll compute errors on GPU too
    
    free(h_nodes_a);
    free(h_nodes_bias);
    free(h_edges_src);
    free(h_edges_dst);
    free(h_edges_w);
    
    printf("[mc_gpu] Propagation completed on GPU\n");
}

// MC Function: GPU-accelerated error computation
void mc_gpu_compute_error(Brain *g, uint64_t node_id) {
    if (!gpu_initialized) return;
    
    uint64_t n = g->header->num_nodes;
    
    // Copy actual activations to GPU
    float *h_nodes_a = (float*)malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) {
        h_nodes_a[i] = g->nodes[i].a;
    }
    cudaMemcpy(d_nodes_a, h_nodes_a, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch error kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    compute_error_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_nodes_a, d_predicted_a, d_node_error, n
    );
    
    cudaDeviceSynchronize();
    
    // Copy errors back to host (if needed)
    // For now, weight updates will happen on GPU
    
    free(h_nodes_a);
    printf("[mc_gpu] Error computation completed on GPU\n");
}

// MC Function: GPU-accelerated weight updates
void mc_gpu_update_edges(Brain *g, uint64_t node_id) {
    if (!gpu_initialized) return;
    
    uint64_t e = g->header->num_edges;
    uint64_t n = g->header->num_nodes;
    float eta = 0.001f; // Learning rate
    
    // Launch update kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (e + threadsPerBlock - 1) / threadsPerBlock;
    update_edges_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_nodes_a, d_node_error,
        d_edges_src, d_edges_dst, d_edges_w,
        e, n, eta
    );
    
    cudaDeviceSynchronize();
    
    // Copy updated weights back to host
    float *h_edges_w = (float*)malloc(e * sizeof(float));
    cudaMemcpy(h_edges_w, d_edges_w, e * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (uint64_t i = 0; i < e; i++) {
        g->edges[i].w = h_edges_w[i];
    }
    
    free(h_edges_w);
    printf("[mc_gpu] Weight updates completed on GPU\n");
}

// MC Function: Initialize GPU
void mc_gpu_init(Brain *g, uint64_t node_id) {
    if (init_gpu_memory(g)) {
        printf("[mc_gpu] GPU acceleration enabled\n");
    } else {
        printf("[mc_gpu] GPU acceleration not available, using CPU\n");
    }
}
} // extern "C"

