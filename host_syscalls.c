/*
 * host_syscalls.c - Host-side syscall implementations
 * 
 * These are called by machine code in the blob.
 * The blob doesn't know what "clang" or "files" are - it just calls syscalls.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stddef.h>

/* Write text to stdout */
static void host_write_text(const uint8_t *bytes, size_t len) {
    fwrite(bytes, 1, len, stdout);
    fflush(stdout);
}

/* Send motor frame (placeholder) */
static void host_send_motor_frame(const uint8_t *frame, size_t len) {
    (void)frame;
    (void)len;
    /* Would send CAN frame to motors */
}

/* Write file */
static void host_write_file(const char *path, const uint8_t *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(data, 1, len, f);
        fclose(f);
    }
}

/* Read file */
static int host_read_file(const char *path, uint8_t **out_buf, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size < 0) {
        fclose(f);
        return -1;
    }
    
    uint8_t *buf = malloc((size_t)size);
    if (!buf) {
        fclose(f);
        return -1;
    }
    
    size_t read = fread(buf, 1, (size_t)size, f);
    fclose(f);
    
    *out_buf = buf;
    *out_len = read;
    return 0;
}

/* Run C compiler */
static int host_run_cc(const char *src_path, const char *out_path) {
    char cmd[1024];
    /* Compile to raw binary (no linking, just object code) */
    snprintf(cmd, sizeof(cmd), "clang -O2 -c -fPIC %s -o %s.o 2>&1", src_path, out_path);
    int ret = system(cmd);
    
    if (ret != 0) return -1;
    
    /* Extract .text section to raw binary */
    char objcopy_cmd[1024];
    snprintf(objcopy_cmd, sizeof(objcopy_cmd), "objcopy -O binary --only-section=.text %s.o %s 2>&1", out_path, out_path);
    ret = system(objcopy_cmd);
    
    return (ret == 0) ? 0 : -1;
}

/* GPU compute (host handles driver - CUDA/Metal/OpenCL) */
static int host_gpu_compute(const GPUComputeRequest *req) {
    if (!req) return -1;
    
    /* TODO: Implement based on available GPU API */
    /* 
     * Options:
     * - CUDA: cuModuleLoad, cuLaunchKernel
     * - Metal: MTLDevice, MTLComputePipelineState
     * - OpenCL: clCreateProgramWithSource, clEnqueueNDRangeKernel
     * 
     * For now, placeholder that copies input to output (CPU fallback)
     */
    if (req->input_data && req->output_data && req->input_data_len <= req->output_data_len) {
        memcpy(req->output_data, req->input_data, req->input_data_len);
        return 0;
    }
    
    return -1;
}

/* Initialize syscall table with host implementations */
void melvin_init_host_syscalls(MelvinSyscalls *syscalls) {
    syscalls->sys_write_text = host_write_text;
    syscalls->sys_send_motor_frame = host_send_motor_frame;
    syscalls->sys_write_file = host_write_file;
    syscalls->sys_read_file = host_read_file;
    syscalls->sys_run_cc = host_run_cc;
    syscalls->sys_gpu_compute = host_gpu_compute;
}

