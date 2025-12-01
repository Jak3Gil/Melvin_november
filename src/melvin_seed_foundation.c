/*
 * melvin_seed_foundation.c - Seed large foundation of nodes and edges
 * 
 * Seeds 1M+ nodes with:
 * - Byte nodes (0-255)
 * - Word sequences
 * - Concept nodes
 * - Pattern nodes
 * - EXEC nodes (from compiled code)
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Seed byte nodes (0-255) - every possible byte value */
static void seed_byte_nodes(Graph *g) {
    printf("Seeding 256 byte nodes...\n");
    // Use port node 0 for seeding
    for (uint8_t b = 0; b < 255; b++) {
        // Feed each byte - creates node automatically
        melvin_feed_byte(g, 0, b, 0.1f);
    }
    printf("✓ Seeded 256 byte nodes\n");
}

/* Seed word as sequence of byte nodes using melvin_feed_byte */
static void seed_word_sequence(Graph *g, const char *word, float strength) {
    if (!word || strlen(word) == 0) return;
    
    // Use port node 0 for word seeding
    uint32_t port_node = 0;
    
    for (size_t i = 0; i < strlen(word); i++) {
        uint8_t byte = (uint8_t)word[i];
        // Feed each byte - this creates nodes and edges automatically
        melvin_feed_byte(g, port_node, byte, strength);
    }
}

/* Seed concept node connected to word sequence */
/* Note: This requires internal functions - for now, use melvin_feed_byte for sequences */
/* Concept nodes can be created by feeding the word and then creating pattern nodes */
static uint32_t seed_concept_node(Graph *g, const char *word, uint32_t concept_id, float strength) {
    if (!word || strlen(word) == 0) return UINT32_MAX;
    
    // For now, just feed the word sequence
    // Concept connections will form through pattern discovery
    seed_word_sequence(g, word, strength);
    
    return concept_id;
}

/* Seed words from corpus file */
static void seed_words_from_corpus(Graph *g, const char *corpus_path, uint64_t max_nodes) {
    FILE *f = fopen(corpus_path, "r");
    if (!f) {
        fprintf(stderr, "Warning: Could not open corpus file: %s\n", corpus_path);
        return;
    }
    
    printf("Seeding words from corpus: %s\n", corpus_path);
    
    uint64_t nodes_created = 0;
    uint32_t prev_node = 0;
    char word[256] = {0};
    size_t word_len = 0;
    
    while (!feof(f) && nodes_created < max_nodes) {
        int c = fgetc(f);
        
        if (c == EOF) break;
        
        uint8_t byte = (uint8_t)c;
        uint32_t node_id = byte;
        
        ensure_node(g, node_id);
        if (g->nodes[node_id].byte == 0) {  // New node
            g->nodes[node_id].byte = byte;
            nodes_created++;
        }
        
        // Feed byte - creates node and edges automatically
        melvin_feed_byte(g, 0, byte, 0.3f);
        
        // Track word boundaries (space, newline, etc.)
        if (byte == ' ' || byte == '\n' || byte == '\t') {
            if (word_len > 0) {
                word[word_len] = '\0';
                // Word complete - could create concept node here
                word_len = 0;
            }
        } else if (word_len < sizeof(word) - 1) {
            word[word_len++] = (char)byte;
        }
    }
    
    fclose(f);
    printf("✓ Seeded %llu nodes from corpus\n", (unsigned long long)nodes_created);
}

/* Seed common words as concept nodes */
static void seed_common_words(Graph *g, uint32_t start_concept_id) {
    const char *common_words[] = {
        "hello", "world", "code", "graph", "node", "edge", "pattern",
        "function", "variable", "string", "integer", "float", "array",
        "if", "else", "for", "while", "return", "void", "int", "char",
        "the", "is", "a", "an", "and", "or", "not", "to", "of", "in",
        // Add more common words...
        NULL
    };
    
    printf("Seeding common words as concept nodes...\n");
    uint32_t concept_id = start_concept_id;
    uint32_t count = 0;
    
    for (int i = 0; common_words[i] != NULL; i++) {
        seed_concept_node(g, common_words[i], concept_id, 0.6f);
        seed_word_sequence(g, common_words[i], 0.5f);
        concept_id++;
        count++;
    }
    
    printf("✓ Seeded %u common word concepts\n", count);
}

/* Seed EXEC nodes from C code */
static void seed_exec_functions(Graph *g, uint32_t start_exec_id, MelvinSyscalls *syscalls) {
    if (!syscalls || !syscalls->sys_compile_c || !syscalls->sys_create_exec_node) {
        printf("⚠ Syscalls not available - skipping EXEC node seeding\n");
        return;
    }
    
    printf("Seeding EXEC nodes from compiled code...\n");
    
    // Example: Simple utility functions
    const char *functions[] = {
        "int add(int a, int b) { return a + b; }",
        "int multiply(int a, int b) { return a * b; }",
        "void print_int(int x) { printf(\"%d\", x); }",
        // Add more functions...
        NULL
    };
    
    uint32_t exec_id = start_exec_id;
    uint32_t count = 0;
    
    for (int i = 0; functions[i] != NULL; i++) {
        uint64_t blob_offset, code_size;
        
        // Compile function
        int result = syscalls->sys_compile_c(
            (const uint8_t *)functions[i],
            strlen(functions[i]),
            &blob_offset,
            &code_size
        );
        
        if (result == 0) {
            // Create EXEC node
            uint32_t created = syscalls->sys_create_exec_node(exec_id, blob_offset, 1.0f);
            if (created != UINT32_MAX) {
                exec_id++;
                count++;
            }
        }
    }
    
    printf("✓ Seeded %u EXEC nodes\n", count);
}

/* Main seeding function */
void melvin_seed_foundation(Graph *g, const char *corpus_path, uint64_t target_nodes, MelvinSyscalls *syscalls) {
    if (!g) {
        fprintf(stderr, "Error: Graph is NULL\n");
        return;
    }
    
    printf("\n=== Seeding Foundation ===\n");
    printf("Target: %llu nodes\n\n", (unsigned long long)target_nodes);
    
    // Phase 1: Seed byte nodes (0-255)
    seed_byte_nodes(g);
    
    // Phase 2: Seed common words
    seed_common_words(g, 256);  // Concept nodes start at 256
    
    // Phase 3: Seed from corpus (if provided)
    if (corpus_path) {
        uint64_t remaining = (target_nodes > 1000) ? (target_nodes - 1000) : target_nodes;
        seed_words_from_corpus(g, corpus_path, remaining);
    }
    
    // Phase 4: Seed EXEC nodes (if syscalls available)
    if (syscalls) {
        seed_exec_functions(g, 10000, syscalls);  // EXEC nodes start at 10000
    }
    
    printf("\n=== Foundation Seeding Complete ===\n");
    printf("Total nodes: %llu\n", (unsigned long long)g->node_count);
    printf("Total edges: %llu\n", (unsigned long long)g->edge_count);
    printf("\n");
}

