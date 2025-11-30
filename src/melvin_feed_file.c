/*
 * melvin_feed_file.c - Feed files/data to Melvin brain
 * 
 * Reads a file and feeds its bytes to Melvin via a port node.
 * Can feed C files, text files, binary data, etc.
 * All data becomes part of the graph's energy landscape.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>

static void feed_file_to_melvin(Graph *g, const char *file_path, uint32_t port_node, float energy) {
    FILE *f = fopen(file_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s: %s\n", file_path, strerror(errno));
        return;
    }
    
    /* Get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (file_size < 0) {
        fprintf(stderr, "Error: Cannot determine size of %s\n", file_path);
        fclose(f);
        return;
    }
    
    printf("Feeding %s (%ld bytes) to port node %u with energy %.3f\n", 
           file_path, file_size, port_node, energy);
    
    uint8_t *buffer = malloc((size_t)file_size);
    if (!buffer) {
        fprintf(stderr, "Error: Out of memory\n");
        fclose(f);
        return;
    }
    
    size_t bytes_read = fread(buffer, 1, (size_t)file_size, f);
    fclose(f);
    
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "Warning: Only read %zu of %ld bytes\n", bytes_read, file_size);
    }
    
    /* Feed each byte to the graph */
    for (size_t i = 0; i < bytes_read; i++) {
        melvin_feed_byte(g, port_node, buffer[i], energy);
    }
    
    /* Trigger propagation after feeding */
    printf("Feeding complete. Triggering UEL propagation...\n");
    melvin_call_entry(g);  /* This runs UEL physics */
    
    free(buffer);
    printf("Done. %zu bytes fed to graph.\n", bytes_read);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <melvin.m file> <file_to_feed> [port_node] [energy]\n", argv[0]);
        fprintf(stderr, "  port_node: Node ID to feed through (default: 0)\n");
        fprintf(stderr, "  energy: Activation energy per byte (default: 0.1)\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s brain.m hello.c 0 0.1\n", argv[0]);
        return 1;
    }
    
    const char *brain_path = argv[1];
    const char *file_path = argv[2];
    uint32_t port_node = (argc > 3) ? (uint32_t)atoi(argv[3]) : 0;
    float energy = (argc > 4) ? (float)atof(argv[4]) : 0.1f;
    
    printf("Opening Melvin brain: %s\n", brain_path);
    Graph *g = melvin_open(brain_path, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", brain_path);
        return 1;
    }
    
    printf("Brain opened: %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    
    feed_file_to_melvin(g, file_path, port_node, energy);
    
    /* Sync to disk */
    melvin_sync(g);
    printf("Brain synced to disk.\n");
    
    melvin_close(g);
    return 0;
}

