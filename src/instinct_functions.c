/*
 * instinct_functions.c - Basic instinct functions for Melvin
 * 
 * These are compiled to machine code and fed into the graph as bytes.
 * The graph learns patterns around these bytes and discovers how to use them.
 * 
 * Core functions:
 * - read_file: Read files (C source, etc.)
 * - compile_code: Compile C code to machine code
 * - understand_machine_code: Learn patterns from machine code bytes
 * - basic_syscalls: Wrappers for syscalls
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

/* Basic file reading - returns file contents as bytes */
uint8_t* instinct_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    
    uint8_t *buf = malloc((size_t)size);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    
    size_t read = fread(buf, 1, (size_t)size, f);
    fclose(f);
    
    if (read != (size_t)size) {
        free(buf);
        return NULL;
    }
    
    *out_len = (size_t)size;
    return buf;
}

/* Basic code compilation - compiles C source to object file */
int instinct_compile_code(const char *src_path, const char *out_path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "gcc -c -o %s %s 2>&1", out_path, src_path);
    return system(cmd);
}

/* Basic machine code understanding - extracts .text section from object file */
uint8_t* instinct_extract_machine_code(const char *obj_path, size_t *out_len) {
    /* Use objcopy or otool to extract .text section */
    char cmd[1024];
    char temp_bin[256];
    snprintf(temp_bin, sizeof(temp_bin), "/tmp/melvin_instinct_%d.bin", getpid());
    
    #ifdef __APPLE__
    /* macOS: use otool + dd */
    snprintf(cmd, sizeof(cmd), 
             "otool -t %s | tail -n +2 | xxd -r -p > %s 2>/dev/null", 
             obj_path, temp_bin);
    #else
    /* Linux: use objcopy */
    snprintf(cmd, sizeof(cmd), 
             "objcopy -O binary --only-section=.text %s %s 2>/dev/null", 
             obj_path, temp_bin);
    #endif
    
    if (system(cmd) != 0) {
        return NULL;
    }
    
    /* Read extracted binary */
    FILE *f = fopen(temp_bin, "rb");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size < 0) {
        fclose(f);
        unlink(temp_bin);
        return NULL;
    }
    
    uint8_t *buf = malloc((size_t)size);
    if (!buf) {
        fclose(f);
        unlink(temp_bin);
        return NULL;
    }
    
    size_t read = fread(buf, 1, (size_t)size, f);
    fclose(f);
    unlink(temp_bin);
    
    if (read != (size_t)size) {
        free(buf);
        return NULL;
    }
    
    *out_len = (size_t)size;
    return buf;
}

/* Basic syscall wrappers */
void instinct_sys_write_text(const uint8_t *bytes, size_t len) {
    fwrite(bytes, 1, len, stdout);
    fflush(stdout);
}

int instinct_sys_read_file(const char *path, uint8_t **out_buf, size_t *out_len) {
    uint8_t *buf = instinct_read_file(path, out_len);
    if (!buf) return -1;
    *out_buf = buf;
    return 0;
}

int instinct_sys_compile(const char *src_path, const char *out_path) {
    return instinct_compile_code(src_path, out_path);
}

/* Entry point - graph can call this */
void instinct_main(void) {
    /* This is where the graph's learned patterns would call instinct functions */
    /* Initially empty - graph learns to populate it */
}

