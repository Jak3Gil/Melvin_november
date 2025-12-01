/*
 * melvin_load_patterns.c - Load patterns from data files (data-driven seeding)
 * 
 * Patterns are defined as sequences of text/bytes. The system:
 * 1. Feeds sequences through melvin_feed_byte (creates nodes naturally)
 * 2. Creates edges between nodes in sequence
 * 3. No hardcoded node IDs - nodes discovered through data flow
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

/* Pattern definition: sequence of tokens that form a pattern */
typedef struct {
    char **tokens;      /* Array of token strings */
    size_t token_count; /* Number of tokens */
    float strength;     /* Edge strength for this pattern */
} PatternDef;

/* Forward declarations */
static int is_arrow(const char *p);
static uint32_t find_pattern_edge(Graph *g, uint32_t src, uint32_t dst);
static uint32_t create_pattern_edge(Graph *g, uint32_t src, uint32_t dst, float w);
static void feed_pattern_sequence(Graph *g, PatternDef *pattern, float strength);

/* Helper: Check if character is arrow (→ or ->) */
static int is_arrow(const char *p) {
    if (!p) return 0;
    if (p[0] == '-' && p[1] == '>') return 2;  /* -> */
    /* Check for UTF-8 arrow → (0xE2 0x86 0x92) */
    if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x86 && (unsigned char)p[2] == 0x92) return 3;
    return 0;
}

/* Feed a pattern sequence: tokens separated by arrows
 * Each token is fed as bytes, creating nodes naturally
 * Edges form between consecutive tokens in the sequence
 */
static void feed_pattern_sequence(Graph *g, PatternDef *pattern, float strength) {
    if (!g || !pattern || pattern->token_count < 2) return;
    
    /* Use a dedicated port node for pattern seeding */
    /* melvin_feed_byte will ensure the node exists */
    uint32_t pattern_port = 1000;  /* Port node for patterns */
    melvin_feed_byte(g, pattern_port, 0, 0.0f);  /* Ensure port node exists */
    
    /* Track the first byte node of each token to create edges between tokens */
    uint32_t *token_first_nodes = malloc(pattern->token_count * sizeof(uint32_t));
    if (!token_first_nodes) return;
    
    /* First pass: Feed all tokens, track first byte node of each */
    for (size_t i = 0; i < pattern->token_count; i++) {
        const char *token = pattern->tokens[i];
        if (!token || strlen(token) == 0) {
            token_first_nodes[i] = UINT32_MAX;
            continue;
        }
        
        /* Feed first byte to get node ID */
        uint8_t first_byte = (uint8_t)token[0];
        melvin_feed_byte(g, pattern_port, first_byte, strength * 0.3f);
        token_first_nodes[i] = (uint32_t)first_byte;
        
        /* Feed remaining bytes of token */
        for (size_t j = 1; j < strlen(token); j++) {
            uint8_t byte = (uint8_t)token[j];
            melvin_feed_byte(g, pattern_port, byte, strength * 0.3f);
        }
    }
    
    /* Second pass: Create edges between consecutive tokens */
    for (size_t i = 0; i < pattern->token_count - 1; i++) {
        uint32_t src_node = token_first_nodes[i];
        uint32_t dst_node = token_first_nodes[i + 1];
        
        if (src_node != UINT32_MAX && dst_node != UINT32_MAX) {
            uint32_t eid = find_pattern_edge(g, src_node, dst_node);
            if (eid == UINT32_MAX) {
                create_pattern_edge(g, src_node, dst_node, strength);
            } else {
                /* Strengthen existing edge */
                if (strength > g->edges[eid].w) {
                    g->edges[eid].w = strength;
                }
            }
        }
    }
    
    free(token_first_nodes);
}

/* Parse pattern line: "TOKEN1 → TOKEN2 → TOKEN3" */
static PatternDef parse_pattern_line(const char *line) {
    PatternDef pattern = {0};
    
    if (!line) return pattern;
    
    /* Skip leading whitespace */
    while (*line && isspace(*line)) line++;
    if (*line == '\0' || *line == '#') return pattern;
    
    /* Count tokens (separated by → or -> or spaces) */
    size_t token_capacity = 16;
    pattern.tokens = malloc(token_capacity * sizeof(char*));
    if (!pattern.tokens) return pattern;
    
    const char *start = line;
    const char *p = line;
    
    while (*p) {
        /* Skip whitespace */
        while (*p && isspace(*p)) p++;
        if (*p == '\0' || *p == '#') break;
        
        start = p;
        
        /* Find end of token (arrow, whitespace, or comment) */
        while (*p && !is_arrow(p) && !isspace(*p) && *p != '#') {
            p++;
        }
        
        if (p > start) {
            /* Allocate and copy token */
            size_t token_len = p - start;
            char *token = malloc(token_len + 1);
            if (token) {
                memcpy(token, start, token_len);
                token[token_len] = '\0';
                
                if (pattern.token_count >= token_capacity) {
                    token_capacity *= 2;
                    pattern.tokens = realloc(pattern.tokens, token_capacity * sizeof(char*));
                }
                
                if (pattern.tokens) {
                    pattern.tokens[pattern.token_count++] = token;
                } else {
                    free(token);
                }
            }
        }
        
        /* Skip arrow */
        int arrow_len = is_arrow(p);
        if (arrow_len > 0) {
            p += arrow_len;
        }
    }
    
    return pattern;
}

/* Free pattern definition */
static void free_pattern(PatternDef *pattern) {
    if (!pattern) return;
    if (pattern->tokens) {
        for (size_t i = 0; i < pattern->token_count; i++) {
            free(pattern->tokens[i]);
        }
        free(pattern->tokens);
    }
    pattern->tokens = NULL;
    pattern->token_count = 0;
}

/* Load patterns from file and create edges */
static void load_patterns_from_file(Graph *g, const char *pattern_file, float default_strength) {
    if (!g || !pattern_file) return;
    
    FILE *f = fopen(pattern_file, "r");
    if (!f) {
        fprintf(stderr, "Warning: Could not open pattern file: %s\n", pattern_file);
        return;
    }
    
    printf("Loading patterns from %s...\n", pattern_file);
    
    char line[1024];
    size_t pattern_count = 0;
    
    while (fgets(line, sizeof(line), f)) {
        /* Skip empty lines and comments */
        if (line[0] == '\n' || line[0] == '#' || line[0] == '\0') continue;
        
        /* Remove trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        /* Parse pattern */
        PatternDef pattern = parse_pattern_line(line);
        
        if (pattern.token_count >= 2) {
            /* Feed pattern sequence - creates nodes and edges naturally */
            pattern.strength = default_strength;
            feed_pattern_sequence(g, &pattern, default_strength);
            pattern_count++;
        }
        
        free_pattern(&pattern);
    }
    
    fclose(f);
    printf("✓ Loaded %zu patterns from %s\n", pattern_count, pattern_file);
}

/* Helper: Find edge between two nodes */
static uint32_t find_pattern_edge(Graph *g, uint32_t src, uint32_t dst) {
    if (!g || src >= g->node_count) return UINT32_MAX;
    uint32_t eid = g->nodes[src].first_out;
    uint32_t max_iter = (uint32_t)(g->edge_count + 1);
    uint32_t iter = 0;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
        if (g->edges[eid].dst == dst) return eid;
        eid = g->edges[eid].next_out;
        iter++;
    }
    return UINT32_MAX;
}

/* Helper: Create edge between two nodes */
static uint32_t create_pattern_edge(Graph *g, uint32_t src, uint32_t dst, float w) {
    if (!g || !g->hdr || src >= g->node_count || dst >= g->node_count) return UINT32_MAX;
    if (g->edge_count >= UINT32_MAX) return UINT32_MAX;
    
    /* Ensure nodes exist by feeding a byte (this calls ensure_node internally) */
    melvin_feed_byte(g, src, 0, 0.0f);
    melvin_feed_byte(g, dst, 0, 0.0f);
    
    /* Check again after ensuring nodes */
    if (src >= g->node_count || dst >= g->node_count) return UINT32_MAX;
    
    uint32_t eid = (uint32_t)g->edge_count++;
    g->hdr->edge_count = g->edge_count;
    Edge *e = &g->edges[eid];
    e->src = src;
    e->dst = dst;
    e->w = w;
    e->next_out = g->nodes[src].first_out;
    e->next_in = g->nodes[dst].first_in;
    g->nodes[src].first_out = eid;
    g->nodes[dst].first_in = eid;
    g->nodes[src].out_degree++;
    g->nodes[dst].in_degree++;
    
    return eid;
}

/* Load patterns and create edges by feeding sequences */
void melvin_load_patterns(Graph *g, const char *pattern_file, float strength) {
    if (!g) {
        fprintf(stderr, "Error: Graph is NULL\n");
        return;
    }
    
    if (!pattern_file) {
        fprintf(stderr, "Error: Pattern file path is NULL\n");
        return;
    }
    
    load_patterns_from_file(g, pattern_file, strength);
}


