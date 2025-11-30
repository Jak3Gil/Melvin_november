/*
 * melvin_seed_instincts.c - Seed basic interaction patterns (instincts)
 * 
 * Seeds the graph with basic patterns that allow it to:
 * - Use syscalls (CPU/GPU/OS interaction)
 * - Read/write files
 * - Use GPU compute
 * - Basic control flow
 * 
 * These are "instincts" - pre-seeded patterns that bias behavior
 * but are subject to UEL modification and improvement.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Helper: Find existing edge */
static uint32_t find_edge(Graph *g, uint32_t src, uint32_t dst) {
    if (src >= g->node_count) return UINT32_MAX;
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

/* Helper: Create edge (same logic as melvin.c) */
static uint32_t create_edge(Graph *g, uint32_t src, uint32_t dst, float w) {
    if (src >= g->node_count || dst >= g->node_count) return UINT32_MAX;
    if (g->edge_count >= UINT32_MAX) return UINT32_MAX;
    
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

/* Helper: Find or create edge between two nodes */
static uint32_t find_or_create_edge(Graph *g, uint32_t src, uint32_t dst, float weight) {
    uint32_t eid = find_edge(g, src, dst);
    if (eid != UINT32_MAX) {
        /* Update weight if stronger */
        if (weight > g->edges[eid].w) {
            g->edges[eid].w = weight;
        }
        return eid;
    }
    return create_edge(g, src, dst, weight);
}

/* Seed a pattern: sequence of node IDs with connections */
static void seed_pattern(Graph *g, uint32_t *nodes, size_t count, float pattern_strength) {
    if (!g || !nodes || count < 2) return;
    
    /* Create sequential connections (pattern flow) */
    for (size_t i = 0; i < count - 1; i++) {
        if (nodes[i] < g->node_count && nodes[i+1] < g->node_count) {
            find_or_create_edge(g, nodes[i], nodes[i+1], pattern_strength);
        }
    }
    
    /* Create bidirectional connections (pattern coherence) */
    for (size_t i = 0; i < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            if (nodes[i] < g->node_count && nodes[j] < g->node_count) {
                /* Weaker bidirectional connection for coherence */
                find_or_create_edge(g, nodes[i], nodes[j], pattern_strength * 0.3f);
                find_or_create_edge(g, nodes[j], nodes[i], pattern_strength * 0.3f);
            }
        }
    }
}

/* Seed MINIMAL bootstrap patterns - just enough to read, compile, and understand */
static void seed_bootstrap_patterns(Graph *g) {
    if (!g || g->node_count < 300) {
        fprintf(stderr, "Warning: Graph too small for bootstrap patterns (need >= 300 nodes, have %llu)\n",
                (unsigned long long)g->node_count);
        return;
    }
    
    /* Bootstrap nodes: minimal set for self-compilation */
    uint32_t NODE_FILE = 256;
    uint32_t NODE_READ = 257;
    uint32_t NODE_DATA = 258;
    uint32_t NODE_COMPILE = 259;
    uint32_t NODE_MACHINE_CODE = 260;
    uint32_t NODE_UNDERSTAND = 261;
    uint32_t NODE_SYSCALL = 262;
    uint32_t NODE_CALL = 263;
    uint32_t NODE_SOURCE = 264;
    uint32_t NODE_BYTES = 265;
    uint32_t NODE_COLD_DATA = 266;  /* Cold data region */
    uint32_t NODE_COPY = 267;        /* Copy operation */
    uint32_t NODE_LEARN = 268;       /* Learning/curiosity */
    
    float bootstrap_strength = 0.6f;  /* Strong enough to bootstrap */
    
    /* CRITICAL PATTERN: File → Read → Data → Compile → Machine Code → Understand */
    uint32_t bootstrap[] = {NODE_FILE, NODE_READ, NODE_DATA, NODE_COMPILE, 
                           NODE_MACHINE_CODE, NODE_UNDERSTAND};
    seed_pattern(g, bootstrap, 6, bootstrap_strength);
    
    /* Pattern: Source → Compile → Machine Code */
    uint32_t compile_pattern[] = {NODE_SOURCE, NODE_COMPILE, NODE_MACHINE_CODE};
    seed_pattern(g, compile_pattern, 3, bootstrap_strength);
    
    /* Pattern: Machine Code → Bytes → Understand */
    uint32_t understand_pattern[] = {NODE_MACHINE_CODE, NODE_BYTES, NODE_UNDERSTAND};
    seed_pattern(g, understand_pattern, 3, bootstrap_strength);
    
    /* Pattern: Syscall → Call → Action */
    uint32_t NODE_ACTION = 275;  /* From seed_syscall_patterns */
    uint32_t syscall_pattern[] = {NODE_SYSCALL, NODE_CALL, NODE_ACTION};
    seed_pattern(g, syscall_pattern, 3, bootstrap_strength);
    
    /* Connect: Read → Source (C files are source) */
    find_or_create_edge(g, NODE_READ, NODE_SOURCE, bootstrap_strength * 0.7f);
    
    /* Connect: Compile → Syscall (compilation uses syscalls) */
    find_or_create_edge(g, NODE_COMPILE, NODE_SYSCALL, bootstrap_strength * 0.6f);
    
    /* CRITICAL: Self-directed learning pattern - Cold Data → Copy → Blob → Read → Learn */
    uint32_t learning_pattern[] = {NODE_COLD_DATA, NODE_COPY, NODE_BYTES, NODE_READ, NODE_LEARN};
    seed_pattern(g, learning_pattern, 5, bootstrap_strength * 0.5f);  /* Weaker - let graph discover */
    
    /* Connect: Learn → Understand (learning leads to understanding) */
    find_or_create_edge(g, NODE_LEARN, NODE_UNDERSTAND, bootstrap_strength * 0.6f);
    
    /* Connect: Curiosity/Exploration → Cold Data (internal drive triggers reading) */
    /* This will be discovered by the graph's exploration drive */
    
    /* Mark key nodes for exploration */
    if (g->output_propensity) {
        if (NODE_COMPILE < g->node_count) g->output_propensity[NODE_COMPILE] = 0.8f;
        if (NODE_UNDERSTAND < g->node_count) g->output_propensity[NODE_UNDERSTAND] = 0.7f;
        if (NODE_CALL < g->node_count) g->output_propensity[NODE_CALL] = 0.7f;
        if (NODE_LEARN < g->node_count) g->output_propensity[NODE_LEARN] = 0.6f;  /* Curiosity drive */
        if (NODE_COPY < g->node_count) g->output_propensity[NODE_COPY] = 0.5f;  /* Moderate exploration */
    }
    
    printf("Seeded bootstrap patterns (read → compile → understand)\n");
}

/* Seed syscall interaction patterns */
static void seed_syscall_patterns(Graph *g) {
    if (!g || g->node_count < 300) {
        fprintf(stderr, "Warning: Graph too small for instincts (need >= 300 nodes, have %llu)\n",
                (unsigned long long)g->node_count);
        return;
    }
    
    /* Reserve node IDs for instinct concepts (256-299) */
    /* 256-259: Syscall concepts */
    uint32_t NODE_SYSCALL = 256;
    uint32_t NODE_WRITE_TEXT = 257;
    uint32_t NODE_READ_FILE = 258;
    uint32_t NODE_GPU_COMPUTE = 259;
    
    /* 260-263: File operations */
    uint32_t NODE_FILE = 260;
    uint32_t NODE_READ = 261;
    uint32_t NODE_WRITE = 262;
    uint32_t NODE_PATH = 263;
    
    /* 264-267: GPU concepts */
    uint32_t NODE_GPU = 264;
    uint32_t NODE_COMPUTE = 265;
    uint32_t NODE_KERNEL = 266;
    uint32_t NODE_DATA = 267;
    
    /* 268-271: Control flow */
    uint32_t NODE_IF = 268;
    uint32_t NODE_THEN = 269;
    uint32_t NODE_ELSE = 270;
    uint32_t NODE_LOOP = 271;
    
    /* 272-275: Output concepts */
    uint32_t NODE_OUTPUT = 272;
    uint32_t NODE_RESULT = 273;
    uint32_t NODE_RESPONSE = 274;
    uint32_t NODE_ACTION = 275;
    
    /* 276-279: Input/Processing concepts */
    uint32_t NODE_INPUT = 276;
    uint32_t NODE_PROCESS = 277;
    uint32_t NODE_TRANSFORM = 278;
    uint32_t NODE_COMPLETE = 279;
    
    /* 280-283: CPU/OS concepts */
    uint32_t NODE_CPU = 280;
    uint32_t NODE_OS = 281;
    uint32_t NODE_EXEC = 282;
    uint32_t NODE_CALL = 283;
    
    /* Initialize instinct nodes (ensure they exist and are initialized) */
    uint32_t instinct_nodes[] = {
        NODE_SYSCALL, NODE_WRITE_TEXT, NODE_READ_FILE, NODE_GPU_COMPUTE,
        NODE_FILE, NODE_READ, NODE_WRITE, NODE_PATH,
        NODE_GPU, NODE_COMPUTE, NODE_KERNEL, NODE_DATA,
        NODE_IF, NODE_THEN, NODE_ELSE, NODE_LOOP,
        NODE_OUTPUT, NODE_RESULT, NODE_RESPONSE, NODE_ACTION,
        NODE_INPUT, NODE_PROCESS, NODE_TRANSFORM, NODE_COMPLETE,
        NODE_CPU, NODE_OS, NODE_EXEC, NODE_CALL
    };
    
    for (size_t i = 0; i < sizeof(instinct_nodes)/sizeof(instinct_nodes[0]); i++) {
        if (instinct_nodes[i] < g->node_count) {
            g->nodes[instinct_nodes[i]].a = 0.0f;
            g->nodes[instinct_nodes[i]].byte = 0;
        }
    }
    
    float instinct_strength = 0.5f;  /* Moderate strength - can be modified by UEL */
    float strong_strength = 0.7f;    /* Stronger for core workflows */
    
    /* ========================================================================
     * CORE WORKFLOW PATTERNS (Input → Process → Output)
     * ======================================================================== */
    
    /* Pattern 1: Input → Process → Output (basic workflow) */
    uint32_t workflow1[] = {NODE_INPUT, NODE_PROCESS, NODE_OUTPUT};
    seed_pattern(g, workflow1, 3, strong_strength);
    
    /* Pattern 2: Input → Transform → Result → Output */
    uint32_t workflow2[] = {NODE_INPUT, NODE_TRANSFORM, NODE_RESULT, NODE_OUTPUT};
    seed_pattern(g, workflow2, 4, instinct_strength);
    
    /* ========================================================================
     * SYSCALL PATTERNS (How to interact with CPU/GPU/OS)
     * ======================================================================== */
    
    /* Pattern 3: Syscall → Write Text → Output */
    uint32_t syscall1[] = {NODE_SYSCALL, NODE_WRITE_TEXT, NODE_OUTPUT};
    seed_pattern(g, syscall1, 3, strong_strength);
    
    /* Pattern 4: File → Read → Data → Process */
    uint32_t syscall2[] = {NODE_FILE, NODE_READ, NODE_DATA, NODE_PROCESS};
    seed_pattern(g, syscall2, 4, strong_strength);
    
    /* Pattern 5: Process → Write → File → Output */
    uint32_t syscall3[] = {NODE_PROCESS, NODE_WRITE, NODE_FILE, NODE_OUTPUT};
    seed_pattern(g, syscall3, 4, instinct_strength);
    
    /* Pattern 6: GPU → Compute → Result → Output */
    uint32_t syscall4[] = {NODE_GPU, NODE_COMPUTE, NODE_RESULT, NODE_OUTPUT};
    seed_pattern(g, syscall4, 4, strong_strength);
    
    /* Pattern 7: Data → GPU → Kernel → Compute → Result */
    uint32_t syscall5[] = {NODE_DATA, NODE_GPU, NODE_KERNEL, NODE_COMPUTE, NODE_RESULT};
    seed_pattern(g, syscall5, 5, instinct_strength);
    
    /* Pattern 8: CPU → Exec → Call → Action */
    uint32_t syscall6[] = {NODE_CPU, NODE_EXEC, NODE_CALL, NODE_ACTION};
    seed_pattern(g, syscall6, 4, instinct_strength);
    
    /* Pattern 9: OS → Syscall → Action → Response */
    uint32_t syscall7[] = {NODE_OS, NODE_SYSCALL, NODE_ACTION, NODE_RESPONSE};
    seed_pattern(g, syscall7, 4, instinct_strength);
    
    /* ========================================================================
     * CONTROL FLOW PATTERNS
     * ======================================================================== */
    
    /* Pattern 10: If → Then → Action */
    uint32_t control1[] = {NODE_IF, NODE_THEN, NODE_ACTION};
    seed_pattern(g, control1, 3, instinct_strength);
    
    /* Pattern 11: If → Else → Action */
    uint32_t control2[] = {NODE_IF, NODE_ELSE, NODE_ACTION};
    seed_pattern(g, control2, 3, instinct_strength);
    
    /* Pattern 12: Loop → Action → Loop (recursive) */
    uint32_t control3[] = {NODE_LOOP, NODE_ACTION, NODE_LOOP};
    seed_pattern(g, control3, 3, instinct_strength);
    
    /* Pattern 13: Loop → Process → Complete → Loop */
    uint32_t control4[] = {NODE_LOOP, NODE_PROCESS, NODE_COMPLETE, NODE_LOOP};
    seed_pattern(g, control4, 4, instinct_strength);
    
    /* ========================================================================
     * CROSS-PATTERN CONNECTIONS (Higher-level patterns)
     * ======================================================================== */
    
    /* Connect syscalls to workflows */
    find_or_create_edge(g, NODE_READ_FILE, NODE_INPUT, instinct_strength * 0.6f);
    find_or_create_edge(g, NODE_WRITE_TEXT, NODE_OUTPUT, instinct_strength * 0.8f);
    find_or_create_edge(g, NODE_GPU_COMPUTE, NODE_PROCESS, instinct_strength * 0.7f);
    find_or_create_edge(g, NODE_RESULT, NODE_OUTPUT, instinct_strength * 0.8f);
    
    /* Connect control flow to workflows */
    find_or_create_edge(g, NODE_IF, NODE_PROCESS, instinct_strength * 0.5f);
    find_or_create_edge(g, NODE_LOOP, NODE_PROCESS, instinct_strength * 0.5f);
    find_or_create_edge(g, NODE_ACTION, NODE_OUTPUT, instinct_strength * 0.7f);
    
    /* Connect CPU/OS to syscalls */
    find_or_create_edge(g, NODE_CPU, NODE_SYSCALL, instinct_strength * 0.6f);
    find_or_create_edge(g, NODE_OS, NODE_SYSCALL, instinct_strength * 0.6f);
    find_or_create_edge(g, NODE_EXEC, NODE_CALL, instinct_strength * 0.7f);
    
    /* Output → Response (feedback loop) */
    find_or_create_edge(g, NODE_OUTPUT, NODE_RESPONSE, instinct_strength * 0.6f);
    find_or_create_edge(g, NODE_RESPONSE, NODE_INPUT, instinct_strength * 0.4f);  /* Feedback */
    
    /* Mark output nodes for exploration drive (high output propensity) */
    if (g->output_propensity) {
        if (NODE_OUTPUT < g->node_count) g->output_propensity[NODE_OUTPUT] = 0.9f;
        if (NODE_RESPONSE < g->node_count) g->output_propensity[NODE_RESPONSE] = 0.8f;
        if (NODE_ACTION < g->node_count) g->output_propensity[NODE_ACTION] = 0.7f;
        if (NODE_WRITE_TEXT < g->node_count) g->output_propensity[NODE_WRITE_TEXT] = 0.8f;
    }
    
    printf("Seeded %zu instinct patterns with %zu cross-connections\n", 
           (size_t)13, (size_t)12);
}

/* Seed byte-to-concept mappings (ASCII patterns) */
static void seed_byte_concept_patterns(Graph *g) {
    if (!g || g->node_count < 256) return;
    
    float byte_strength = 0.2f;  /* Weak connections - graph learns to strengthen them */
    
    /* Connect common ASCII bytes to instinct concepts */
    /* 's' (115) → syscall concept */
    if (115 < g->node_count && 256 < g->node_count) {
        find_or_create_edge(g, 115, 256, byte_strength);  /* 's' → SYSCALL */
    }
    
    /* 'f' (102) → file concept */
    if (102 < g->node_count && 260 < g->node_count) {
        find_or_create_edge(g, 102, 260, byte_strength);  /* 'f' → FILE */
    }
    
    /* 'g' (103) → GPU concept */
    if (103 < g->node_count && 264 < g->node_count) {
        find_or_create_edge(g, 103, 264, byte_strength);  /* 'g' → GPU */
    }
    
    /* 'r' (114) → read concept */
    if (114 < g->node_count && 261 < g->node_count) {
        find_or_create_edge(g, 114, 261, byte_strength);  /* 'r' → READ */
    }
    
    /* 'w' (119) → write concept */
    if (119 < g->node_count && 262 < g->node_count) {
        find_or_create_edge(g, 119, 262, byte_strength);  /* 'w' → WRITE */
    }
    
    /* 'c' (99) → compute/CPU concept */
    if (99 < g->node_count && 265 < g->node_count) {
        find_or_create_edge(g, 99, 265, byte_strength);  /* 'c' → COMPUTE */
    }
    if (99 < g->node_count && 280 < g->node_count) {
        find_or_create_edge(g, 99, 280, byte_strength);  /* 'c' → CPU */
    }
    
    /* Common word patterns: "sys", "file", "gpu", "read", "write", "exec" */
    /* 's' → 'y' → 's' (sys) */
    if (115 < g->node_count && 121 < g->node_count) {
        find_or_create_edge(g, 115, 121, byte_strength * 0.5f);  /* 's' → 'y' */
    }
    if (121 < g->node_count && 115 < g->node_count) {
        find_or_create_edge(g, 121, 115, byte_strength * 0.5f);  /* 'y' → 's' */
    }
    
    /* 'f' → 'i' → 'l' → 'e' (file) */
    if (102 < g->node_count && 105 < g->node_count) {
        find_or_create_edge(g, 102, 105, byte_strength * 0.5f);  /* 'f' → 'i' */
    }
    if (105 < g->node_count && 108 < g->node_count) {
        find_or_create_edge(g, 105, 108, byte_strength * 0.5f);  /* 'i' → 'l' */
    }
    if (108 < g->node_count && 101 < g->node_count) {
        find_or_create_edge(g, 108, 101, byte_strength * 0.5f);  /* 'l' → 'e' */
    }
    
    printf("Seeded byte-to-concept mappings\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <melvin.m file>\n", argv[0]);
        return 1;
    }
    
    const char *path = argv[1];
    
    /* Open existing .m file */
    Graph *g = melvin_open(path, 0, 0, 0);  /* 0 = use existing file */
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", path);
        return 1;
    }
    
    printf("Seeding instincts into %s...\n", path);
    printf("Graph has %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    
    /* Seed bootstrap patterns FIRST (minimal set for self-compilation) */
    seed_bootstrap_patterns(g);
    
    /* Then seed full syscall patterns */
    seed_syscall_patterns(g);
    seed_byte_concept_patterns(g);
    
    /* Sync to disk */
    melvin_sync(g);
    
    printf("Instincts seeded. New edge count: %llu\n", 
           (unsigned long long)g->edge_count);
    
    melvin_close(g);
    return 0;
}

