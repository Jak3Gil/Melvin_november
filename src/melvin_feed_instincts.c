/*
 * melvin_feed_instincts.c - Feed instinct machine code bytes into graph
 * 
 * Compiles instinct_functions.c to machine code, extracts bytes,
 * and feeds them into the graph so Melvin can learn to use them.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

/* Extract machine code from compiled object file */
static uint8_t* extract_machine_code(const char *obj_path, size_t *out_len) {
    char temp_bin[256];
    snprintf(temp_bin, sizeof(temp_bin), "/tmp/melvin_extract_%d.bin", getpid());
    
    #ifdef __APPLE__
    /* macOS: use otool + xxd */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), 
             "otool -t %s 2>/dev/null | tail -n +2 | xxd -r -p > %s 2>/dev/null", 
             obj_path, temp_bin);
    if (system(cmd) != 0) {
        return NULL;
    }
    #else
    /* Linux: use objcopy */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), 
             "objcopy -O binary --only-section=.text %s %s 2>/dev/null", 
             obj_path, temp_bin);
    if (system(cmd) != 0) {
        return NULL;
    }
    #endif
    
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

/* Feed machine code bytes into graph */
static void feed_machine_code(Graph *g, uint32_t port_node, const uint8_t *code, size_t len) {
    printf("Feeding %zu bytes of machine code into graph...\n", len);
    
    float energy = 1.0f;  /* High energy for machine code */
    
    for (size_t i = 0; i < len; i++) {
        melvin_feed_byte(g, port_node, code[i], energy);
        
        /* Slight decay for sequential bytes */
        energy *= 0.99f;
    }
    
    printf("Fed machine code bytes. Graph will learn patterns around them.\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <melvin.m file> [instinct_functions.o]\n", argv[0]);
        fprintf(stderr, "  If instinct_functions.o not provided, will compile it first.\n");
        return 1;
    }
    
    const char *melvin_path = argv[1];
    const char *obj_path = (argc > 2) ? argv[2] : NULL;
    
    /* Compile instinct_functions.c if needed */
    if (!obj_path) {
        printf("Compiling instinct_functions.c...\n");
        if (system("gcc -c -o /tmp/melvin_instincts.o src/instinct_functions.c 2>&1") != 0) {
            fprintf(stderr, "Failed to compile instinct_functions.c\n");
            return 1;
        }
        obj_path = "/tmp/melvin_instincts.o";
    }
    
    /* Extract machine code */
    size_t code_len = 0;
    uint8_t *machine_code = extract_machine_code(obj_path, &code_len);
    if (!machine_code || code_len == 0) {
        fprintf(stderr, "Failed to extract machine code from %s\n", obj_path);
        return 1;
    }
    
    printf("Extracted %zu bytes of machine code\n", code_len);
    
    /* Open graph */
    Graph *g = melvin_open(melvin_path, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", melvin_path);
        free(machine_code);
        return 1;
    }
    
    printf("Graph has %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    
    /* Use a dedicated port node for machine code (node 300) */
    uint32_t machine_code_port = 300;
    if (machine_code_port >= g->node_count) {
        fprintf(stderr, "Warning: Graph too small, using node 0 as port\n");
        machine_code_port = 0;
    }
    
    printf("Note: Run 'melvin_seed_instincts' first to seed bootstrap patterns.\n");
    printf("This tool only feeds machine code bytes - patterns should be seeded separately.\n");
    
    /* Feed machine code bytes into graph */
    feed_machine_code(g, machine_code_port, machine_code, code_len);
    
    /* Sync to disk */
    melvin_sync(g);
    
    printf("Machine code fed into graph. Melvin can now learn to use these functions.\n");
    printf("Graph will discover patterns around the machine code bytes.\n");
    
    free(machine_code);
    melvin_close(g);
    return 0;
}

