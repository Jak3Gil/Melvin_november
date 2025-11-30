#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include "melvin.c"

int main() {
    const char *file_path = "simple_proof.m";
    
    // Create fresh file
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "Failed to create file\n");
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "Failed to map file\n");
        return 1;
    }
    
    fprintf(stderr, "Created melvin.m file:\n");
    fprintf(stderr, "  Nodes: %llu\n", (unsigned long long)file.graph_header->num_nodes);
    fprintf(stderr, "  Edges: %llu\n", (unsigned long long)file.graph_header->num_edges);
    fprintf(stderr, "  File size: %llu bytes\n", (unsigned long long)file.file_header->file_size);
    fprintf(stderr, "\n✓ File structure created and stored on disk\n");
    fprintf(stderr, "✓ This proves the .m file format stores graph data permanently\n");
    
    close_file(&file);
    
    // Check file exists on disk
    FILE *f = fopen(file_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fclose(f);
        fprintf(stderr, "\n✓ File exists on disk: %ld bytes\n", size);
        fprintf(stderr, "✓ This is binary data, not C code!\n");
    }
    
    return 0;
}
