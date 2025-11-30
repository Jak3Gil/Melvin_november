/*
 * CUDA Helper for Melvin EXEC
 * 
 * Provides GPU-accelerated computation for EXEC nodes.
 * This is purely an implementation detail - from the graph's perspective,
 * EXEC still returns a scalar that gets converted to energy.
 */

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Forward declaration - must match definition in melvin.c
// For now, we'll just read node activations from a buffer
typedef struct GraphSlice {
    float *node_activations;  // Host pointer to node activations
    uint64_t num_nodes;       // Number of nodes to process
    uint64_t exec_node_id;    // ID of the EXEC node (for context)
} GraphSlice;

// CUDA kernel: sum absolute activations over a window of nodes
__global__ void sum_abs_activations_kernel(const float *activations, float *result, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_nodes) {
        float abs_val = fabsf(activations[idx]);
        atomicAdd(result, abs_val);
    }
}

// CUDA kernel: compute a simple aggregation (sum of |activation|^2)
__global__ void compute_activation_energy_kernel(const float *activations, float *result, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_nodes) {
        float val = activations[idx];
        float energy = val * val;  // |activation|^2
        atomicAdd(result, energy);
    }
}

// Initialize CUDA context (lazy initialization)
static bool cuda_initialized = false;

static void init_cuda_if_needed() {
    if (!cuda_initialized) {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA] Warning: Failed to set device: %s\n", cudaGetErrorString(err));
            cuda_initialized = false;
            return;
        }
        cuda_initialized = true;
    }
}

// Main CUDA EXEC operation
// Returns a scalar (double) that represents the result of GPU computation
extern "C" double melvin_gpu_exec_op(GraphSlice *g, uint64_t exec_node_id) {
    if (!g || !g->node_activations || g->num_nodes == 0) {
        return 0.0;
    }
    
    init_cuda_if_needed();
    if (!cuda_initialized) {
        return 0.0;
    }
    
    // Limit the number of nodes we process (for simplicity)
    uint64_t num_nodes = g->num_nodes;
    if (num_nodes > 1024) {
        num_nodes = 1024;  // Process first 1024 nodes
    }
    
    // Allocate device memory
    float *d_activations = NULL;
    float *d_result = NULL;
    float h_result = 0.0f;
    
    cudaError_t err;
    
    // Allocate and copy activations to device
    err = cudaMalloc((void**)&d_activations, num_nodes * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Failed to allocate device memory for activations: %s\n", 
                cudaGetErrorString(err));
        return 0.0;
    }
    
    err = cudaMemcpy(d_activations, g->node_activations, num_nodes * sizeof(float), 
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Failed to copy activations to device: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_activations);
        return 0.0;
    }
    
    // Allocate result buffer on device
    err = cudaMalloc((void**)&d_result, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Failed to allocate device memory for result: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_activations);
        return 0.0;
    }
    
    // Initialize result to zero
    err = cudaMemset(d_result, 0, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Failed to initialize result: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_activations);
        cudaFree(d_result);
        return 0.0;
    }
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;
    
    // Use the activation energy kernel (sum of |activation|^2)
    compute_activation_energy_kernel<<<num_blocks, threads_per_block>>>(
        d_activations, d_result, (int)num_nodes);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_activations);
        cudaFree(d_result);
        return 0.0;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_activations);
        cudaFree(d_result);
        return 0.0;
    }
    
    // Copy result back to host
    err = cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Failed to copy result from device: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_activations);
        cudaFree(d_result);
        return 0.0;
    }
    
    // Cleanup
    cudaFree(d_activations);
    cudaFree(d_result);
    
    // Return as double (normalized to reasonable range)
    // The result is sum of |activation|^2, which can be large
    // Normalize to [0, 1] range for energy injection
    double normalized = (double)h_result;
    if (normalized > 1000.0) {
        normalized = 1000.0;  // Cap at 1000
    }
    normalized = normalized / 1000.0;  // Normalize to [0, 1]
    
    return normalized;
}

#else  // !HAVE_CUDA

// Stub implementation when CUDA is not available
extern "C" double melvin_gpu_exec_op(void *g, uint64_t exec_node_id) {
    (void)g;
    (void)exec_node_id;
    return 0.0;
}

#endif  // HAVE_CUDA

