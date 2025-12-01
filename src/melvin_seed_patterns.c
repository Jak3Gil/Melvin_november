/*
 * melvin_seed_patterns.c - Seed patterns from data files (data-driven)
 * 
 * Usage: melvin_seed_patterns <melvin.m> [pattern_file] [strength]
 * 
 * Loads patterns from pattern_file (default: corpus/basic/patterns.txt)
 * and creates edges by feeding sequences through the graph.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declaration */
void melvin_load_patterns(Graph *g, const char *pattern_file, float strength);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <melvin.m file> [pattern_file] [strength]\n", argv[0]);
        fprintf(stderr, "  pattern_file: Path to pattern definitions (default: corpus/basic/patterns.txt)\n");
        fprintf(stderr, "  strength: Edge strength (default: 0.6)\n");
        return 1;
    }
    
    const char *path = argv[1];
    const char *pattern_file = (argc >= 3) ? argv[2] : "corpus/basic/patterns.txt";
    float strength = (argc >= 4) ? (float)atof(argv[3]) : 0.6f;
    
    /* Open existing .m file */
    Graph *g = melvin_open(path, 0, 0, 0);  /* 0 = use existing file */
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", path);
        return 1;
    }
    
    printf("Loading patterns into %s...\n", path);
    printf("Pattern file: %s\n", pattern_file);
    printf("Edge strength: %.2f\n", strength);
    printf("Graph has %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    
    /* Load patterns from file */
    melvin_load_patterns(g, pattern_file, strength);
    
    /* Sync to disk */
    melvin_sync(g);
    
    printf("\nPatterns loaded. New edge count: %llu\n", 
           (unsigned long long)g->edge_count);
    
    melvin_close(g);
    return 0;
}

