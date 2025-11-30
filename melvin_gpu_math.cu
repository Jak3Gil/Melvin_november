/*
 * CUDA Math Operations for Melvin EXEC
 * 
 * Provides GPU-accelerated arithmetic operations.
 * The system can learn to route to these GPU EXEC nodes via universal laws.
 */

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Forward declaration
typedef struct GraphSlice {
    float *node_activations;
    uint64_t num_nodes;
    uint64_t exec_node_id;
    // For math operations, we need operands
    float operand_a;
    float operand_b;
    int operation;  // 0=add, 1=multiply, 2=subtract
} GraphSlice;

// CUDA kernel: Vector addition (a + b for each element)
__global__ void gpu_add_kernel(const float *a, const float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel: Vector multiplication (a * b for each element)
__global__ void gpu_multiply_kernel(const float *a, const float *b, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel: Scalar addition (a + b, returns single value)
__global__ void gpu_scalar_add_kernel(float a, float b, float *result) {
    *result = a + b;
}

// CUDA kernel: Scalar multiply (a * b, returns single value)
__global__ void gpu_scalar_multiply_kernel(float a, float b, float *result) {
    *result = a * b;
}

// CUDA kernel: Scalar subtract (a - b, returns single value)
__global__ void gpu_scalar_subtract_kernel(float a, float b, float *result) {
    *result = a - b;
}

// Initialize CUDA context
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

// GPU math operation - performs actual arithmetic
extern "C" double melvin_gpu_math_op(GraphSlice *g, uint64_t exec_node_id, 
                                     float operand_a, float operand_b, int operation) {
    if (!g) {
        return 0.0;
    }
    
    init_cuda_if_needed();
    if (!cuda_initialized) {
        return 0.0;
    }
    
    float *d_result = NULL;
    float h_result = 0.0f;
    cudaError_t err;
    
    // Allocate result buffer on device
    err = cudaMalloc((void**)&d_result, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Failed to allocate device memory: %s\n", 
                cudaGetErrorString(err));
        return 0.0;
    }
    
    // Launch appropriate kernel based on operation
    switch (operation) {
        case 0:  // ADD
            gpu_scalar_add_kernel<<<1, 1>>>(operand_a, operand_b, d_result);
            break;
        case 1:  // MULTIPLY
            gpu_scalar_multiply_kernel<<<1, 1>>>(operand_a, operand_b, d_result);
            break;
        case 2:  // SUBTRACT
            gpu_scalar_subtract_kernel<<<1, 1>>>(operand_a, operand_b, d_result);
            break;
        default:
            fprintf(stderr, "[CUDA] Unknown operation: %d\n", operation);
            cudaFree(d_result);
            return 0.0;
    }
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_result);
        return 0.0;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_result);
        return 0.0;
    }
    
    // Copy result back to host
    err = cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Failed to copy result: %s\n", cudaGetErrorString(err));
        cudaFree(d_result);
        return 0.0;
    }
    
    // Cleanup
    cudaFree(d_result);
    
    // Return result (normalize to [0, 1] for energy injection)
    // For math results, we'll map them to a reasonable range
    double normalized = (double)h_result;
    if (normalized < 0.0) normalized = 0.0;
    if (normalized > 100.0) normalized = 100.0;
    normalized = normalized / 100.0;  // Normalize to [0, 1]
    
    return normalized;
}

#else  // !HAVE_CUDA

// Stub when CUDA not available
extern "C" double melvin_gpu_math_op(void *g, uint64_t exec_node_id,
                                     float operand_a, float operand_b, int operation) {
    (void)g;
    (void)exec_node_id;
    (void)operand_a;
    (void)operand_b;
    (void)operation;
    return 0.0;
}

#endif  // HAVE_CUDA

