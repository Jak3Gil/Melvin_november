/*
 * host_syscalls.c - Host-side syscall implementations
 * 
 * These are called by machine code in the blob.
 * The blob doesn't know what "clang" or "files" are - it just calls syscalls.
 */

#include "melvin.h"
#include "melvin_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stddef.h>
#include <sys/wait.h>

/* Forward declare - defined in melvin.c */
extern Graph* melvin_get_current_graph(void);

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

/* Copy from cold_data to blob (self-directed learning) */
/* Note: This needs Graph* context, which is provided via thread-local storage */
/* The blob calls this syscall, and melvin.c sets up the context */
extern Graph* melvin_get_current_graph(void);  /* Defined in melvin.c */

static void host_copy_from_cold(uint64_t cold_offset, uint64_t length, uint64_t blob_target_offset) {
    Graph *g = melvin_get_current_graph();
    if (!g) return;
    
    melvin_copy_from_cold(g, cold_offset, length, blob_target_offset);
}

/* Tool syscall wrappers - automatically feed outputs into graph */
/* Graph learns from tool outputs through UEL physics */

static int host_llm_generate(const uint8_t *prompt, size_t prompt_len,
                            uint8_t **response, size_t *response_len) {
    int result = melvin_tool_llm_generate(prompt, prompt_len, response, response_len);
    
    /* AUTOMATIC TOOL INTEGRATION: Feed tool output into graph */
    /* Graph learns patterns from tool outputs through UEL */
    if (result == 0 && response && *response && response_len && *response_len > 0) {
        Graph *g = melvin_get_current_graph();
        if (g) {
            /* Feed LLM output (510-519) into graph as new patterns */
            for (size_t i = 0; i < *response_len && i < 100; i++) {
                uint32_t target_node = 510 + (i % 10);  /* LLM output ports */
                if (target_node < g->node_count) {
                    melvin_feed_byte(g, target_node, (*response)[i], 0.4f);  /* Strong - tool output creates patterns */
                }
            }
            /* Also feed to memory for learning */
            for (size_t i = 0; i < *response_len && i < 50; i++) {
                uint32_t mem_node = 201 + (i % 10);  /* Working memory */
                if (mem_node < g->node_count) {
                    melvin_feed_byte(g, mem_node, (*response)[i], 0.3f);
                }
            }
        }
    } else if (result != 0) {
        /* Tool failure - feed error signal to graph */
        Graph *g = melvin_get_current_graph();
        if (g && g->node_count > 250) {
            melvin_feed_byte(g, 250, 1, 0.5f);  /* Error detection node */
        }
    }
    
    return result;
}

static int host_vision_identify(const uint8_t *image_bytes, size_t image_len,
                                uint8_t **labels, size_t *labels_len) {
    int result = melvin_tool_vision_identify(image_bytes, image_len, labels, labels_len);
    
    /* AUTOMATIC TOOL INTEGRATION: Feed vision output into graph */
    if (result == 0 && labels && *labels && labels_len && *labels_len > 0) {
        Graph *g = melvin_get_current_graph();
        if (g) {
            /* Feed vision output (410-419) into graph */
            for (size_t i = 0; i < *labels_len && i < 100; i++) {
                uint32_t target_node = 410 + (i % 10);  /* Vision output ports */
                if (target_node < g->node_count) {
                    melvin_feed_byte(g, target_node, (*labels)[i], 0.4f);
                }
            }
            /* Feed to memory */
            for (size_t i = 0; i < *labels_len && i < 50; i++) {
                uint32_t mem_node = 202 + (i % 10);
                if (mem_node < g->node_count) {
                    melvin_feed_byte(g, mem_node, (*labels)[i], 0.3f);
                }
            }
        }
    } else if (result != 0) {
        /* Tool failure */
        Graph *g = melvin_get_current_graph();
        if (g && g->node_count > 250) {
            melvin_feed_byte(g, 250, 1, 0.5f);  /* Error detection */
        }
    }
    
    return result;
}

static int host_audio_stt(const uint8_t *audio_bytes, size_t audio_len,
                         uint8_t **text, size_t *text_len) {
    int result = melvin_tool_audio_stt(audio_bytes, audio_len, text, text_len);
    
    /* AUTOMATIC TOOL INTEGRATION: Feed STT output into graph */
    if (result == 0 && text && *text && text_len && *text_len > 0) {
        Graph *g = melvin_get_current_graph();
        if (g) {
            /* Feed STT output (310-319) into graph */
            for (size_t i = 0; i < *text_len && i < 100; i++) {
                uint32_t target_node = 310 + (i % 10);  /* STT output ports */
                if (target_node < g->node_count) {
                    melvin_feed_byte(g, target_node, (*text)[i], 0.4f);
                }
            }
            /* Feed to memory */
            for (size_t i = 0; i < *text_len && i < 50; i++) {
                uint32_t mem_node = 200 + (i % 10);
                if (mem_node < g->node_count) {
                    melvin_feed_byte(g, mem_node, (*text)[i], 0.3f);
                }
            }
        }
    } else if (result != 0) {
        /* Tool failure */
        Graph *g = melvin_get_current_graph();
        if (g && g->node_count > 250) {
            melvin_feed_byte(g, 250, 1, 0.5f);  /* Error detection */
        }
    }
    
    return result;
}

static int host_audio_tts(const uint8_t *text, size_t text_len,
                          uint8_t **audio_bytes, size_t *audio_len) {
    int result = melvin_tool_audio_tts(text, text_len, audio_bytes, audio_len);
    
    /* AUTOMATIC TOOL INTEGRATION: Feed TTS output into graph */
    if (result == 0 && audio_bytes && *audio_bytes && audio_len && *audio_len > 0) {
        Graph *g = melvin_get_current_graph();
        if (g) {
            /* Feed TTS output (610-619) into graph */
            for (size_t i = 0; i < *audio_len && i < 100; i++) {
                uint32_t target_node = 610 + (i % 10);  /* TTS output ports */
                if (target_node < g->node_count) {
                    melvin_feed_byte(g, target_node, (*audio_bytes)[i], 0.4f);
                }
            }
            /* Feed to memory */
            for (size_t i = 0; i < *audio_len && i < 50; i++) {
                uint32_t mem_node = 203 + (i % 10);
                if (mem_node < g->node_count) {
                    melvin_feed_byte(g, mem_node, (*audio_bytes)[i], 0.3f);
                }
            }
        }
    } else if (result != 0) {
        /* Tool failure */
        Graph *g = melvin_get_current_graph();
        if (g && g->node_count > 250) {
            melvin_feed_byte(g, 250, 1, 0.5f);  /* Error detection */
        }
    }
    
    return result;
}

/* Compile C source to machine code and store in blob */
static int host_compile_c(const uint8_t *c_source, size_t source_len,
                          uint64_t *blob_offset, uint64_t *code_size) {
    if (!c_source || source_len == 0 || !blob_offset || !code_size) {
        return -1;
    }
    
    Graph *g = melvin_get_current_graph();
    if (!g || !g->blob || g->hdr->blob_size == 0) {
        return -1;
    }
    
    /* Write C source to temporary file */
    char temp_source[] = "/tmp/melvin_blob_XXXXXX.c";
    int fd = mkstemps(temp_source, 2);
    if (fd < 0) {
        return -1;
    }
    
    ssize_t written = write(fd, c_source, source_len);
    close(fd);
    
    if (written != (ssize_t)source_len) {
        unlink(temp_source);
        return -1;
    }
    
    /* Compile to object file */
    char temp_obj[] = "/tmp/melvin_blob_XXXXXX.o";
    int obj_fd = mkstemps(temp_obj, 2);
    if (obj_fd < 0) {
        unlink(temp_source);
        return -1;
    }
    close(obj_fd);
    
    char compile_cmd[512];
    snprintf(compile_cmd, sizeof(compile_cmd),
             "gcc -c -fPIC -o %s %s 2>/dev/null", temp_obj, temp_source);
    
    int compile_result = system(compile_cmd);
    unlink(temp_source);
    
    if (compile_result != 0) {
        unlink(temp_obj);
        return -1;  /* Compilation failed */
    }
    
    /* Read object file and extract .text section */
    /* For now, use objcopy to extract binary */
    char temp_bin[] = "/tmp/melvin_blob_XXXXXX.bin";
    int bin_fd = mkstemps(temp_bin, 4);
    if (bin_fd < 0) {
        unlink(temp_obj);
        return -1;
    }
    close(bin_fd);
    
    char objcopy_cmd[512];
    snprintf(objcopy_cmd, sizeof(objcopy_cmd),
             "objcopy -O binary -j .text %s %s 2>/dev/null", temp_obj, temp_bin);
    
    int objcopy_result = system(objcopy_cmd);
    unlink(temp_obj);
    
    if (objcopy_result != 0) {
        unlink(temp_bin);
        return -1;  /* objcopy failed */
    }
    
    /* Read binary code */
    FILE *bin_file = fopen(temp_bin, "rb");
    if (!bin_file) {
        unlink(temp_bin);
        return -1;
    }
    
    fseek(bin_file, 0, SEEK_END);
    long bin_size = ftell(bin_file);
    fseek(bin_file, 0, SEEK_SET);
    
    if (bin_size <= 0 || bin_size > (long)(g->hdr->blob_size - 256)) {
        fclose(bin_file);
        unlink(temp_bin);
        return -1;  /* Code too large */
    }
    
    /* Find free space in blob (after current main_entry_offset) */
    uint64_t offset = g->hdr->main_entry_offset;
    if (offset == 0) {
        offset = 256;  /* Start after header area */
    } else {
        offset += 256;  /* After existing code */
    }
    
    /* Ensure we have space */
    if (offset + (uint64_t)bin_size > g->hdr->blob_size) {
        fclose(bin_file);
        unlink(temp_bin);
        return -1;  /* Not enough space */
    }
    
    /* Read code into blob */
    size_t read_size = fread(g->blob + offset, 1, (size_t)bin_size, bin_file);
    fclose(bin_file);
    unlink(temp_bin);
    
    if (read_size != (size_t)bin_size) {
        return -1;
    }
    
    /* Feed compiled code into graph as patterns to learn from */
    /* Graph learns from the machine code patterns */
    for (size_t i = 0; i < read_size && i < 1000; i++) {
        uint32_t target_node = 700 + (i % 100);  /* Code pattern nodes (700-799) */
        if (target_node < g->node_count) {
            melvin_feed_byte(g, target_node, g->blob[offset + i], 0.3f);
        }
    }
    
    *blob_offset = offset;
    *code_size = (uint64_t)bin_size;
    
    return 0;
}

/* Initialize syscall table with host implementations */
void melvin_init_host_syscalls(MelvinSyscalls *syscalls) {
    syscalls->sys_write_text = host_write_text;
    syscalls->sys_send_motor_frame = host_send_motor_frame;
    syscalls->sys_write_file = host_write_file;
    syscalls->sys_read_file = host_read_file;
    syscalls->sys_run_cc = host_run_cc;
    syscalls->sys_gpu_compute = host_gpu_compute;
    syscalls->sys_copy_from_cold = host_copy_from_cold;
    
    /* Pattern generation tools */
    syscalls->sys_llm_generate = host_llm_generate;
    syscalls->sys_vision_identify = host_vision_identify;
    syscalls->sys_audio_stt = host_audio_stt;
    syscalls->sys_audio_tts = host_audio_tts;
    syscalls->sys_compile_c = host_compile_c;
}

