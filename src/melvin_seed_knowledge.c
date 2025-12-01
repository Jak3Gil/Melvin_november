/*
 * melvin_seed_knowledge.c - Seed knowledge from corpus files (math, wiki, etc.)
 * 
 * Usage: melvin_seed_knowledge <melvin.m> <corpus_dir> [strength]
 * 
 * Loads all .txt files from corpus_dir and feeds them through the graph
 * to create nodes and edges naturally.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

/* Feed a text file into the graph */
static void feed_text_file(Graph *g, const char *filepath, float energy) {
    if (!g || !filepath) return;
    
    FILE *f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "Warning: Could not open %s\n", filepath);
        return;
    }
    
    printf("Feeding %s...\n", filepath);
    
    uint32_t port_node = 0;  /* Use port 0 for general knowledge */
    size_t byte_count = 0;
    int c;
    
    while ((c = fgetc(f)) != EOF) {
        uint8_t byte = (uint8_t)c;
        melvin_feed_byte(g, port_node, byte, energy);
        byte_count++;
    }
    
    fclose(f);
    printf("  ✓ Fed %zu bytes from %s\n", byte_count, filepath);
}

/* Recursively process directory */
static void process_directory(Graph *g, const char *dirpath, float energy) {
    DIR *dir = opendir(dirpath);
    if (!dir) {
        fprintf(stderr, "Warning: Could not open directory %s\n", dirpath);
        return;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;  /* Skip hidden files */
        
        char fullpath[1024];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
        
        struct stat st;
        if (stat(fullpath, &st) != 0) continue;
        
        if (S_ISDIR(st.st_mode)) {
            /* Recursively process subdirectories */
            process_directory(g, fullpath, energy);
        } else if (S_ISREG(st.st_mode)) {
            /* Check if it's a text file */
            size_t len = strlen(entry->d_name);
            if (len >= 4 && strcmp(entry->d_name + len - 4, ".txt") == 0) {
                feed_text_file(g, fullpath, energy);
            }
        }
    }
    
    closedir(dir);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <melvin.m file> <corpus_dir> [strength]\n", argv[0]);
        fprintf(stderr, "  corpus_dir: Directory containing .txt seed files\n");
        fprintf(stderr, "  strength: Energy per byte (default: 0.3)\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s data/brain.m corpus/math 0.4\n", argv[0]);
        fprintf(stderr, "  %s data/brain.m corpus/wiki 0.3\n", argv[0]);
        return 1;
    }
    
    const char *path = argv[1];
    const char *corpus_dir = argv[2];
    float energy = (argc >= 4) ? (float)atof(argv[3]) : 0.3f;
    
    /* Open existing .m file */
    Graph *g = melvin_open(path, 0, 0, 0);  /* 0 = use existing file */
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", path);
        return 1;
    }
    
    printf("Seeding knowledge into %s...\n", path);
    printf("Corpus directory: %s\n", corpus_dir);
    printf("Energy per byte: %.2f\n", energy);
    printf("Graph has %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    printf("\n");
    
    /* Process all .txt files in corpus directory */
    process_directory(g, corpus_dir, energy);
    
    /* Sync to disk */
    melvin_sync(g);
    
    printf("\nKnowledge seeding complete.\n");
    printf("New node count: %llu\n", (unsigned long long)g->node_count);
    printf("New edge count: %llu\n", (unsigned long long)g->edge_count);
    
    melvin_close(g);
    return 0;
}

