/*
 * Create EXEC Routing Edges
 * 
 * Preseed edges from semantic patterns to EXEC nodes
 * This bootstraps the system so patterns know which EXEC to call
 * 
 * Strategy:
 * 1. Create primitive EXEC nodes (using syscalls - no machine code needed!)
 * 2. Create semantic routing edges (weak, learnable)
 * 3. Let graph strengthen/weaken based on usage
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/melvin.h"

/* EXEC node definitions using syscalls */
typedef struct {
    uint32_t node_id;
    const char *name;
    const char *description;
    float threshold_ratio;  /* How activated to trigger (0.5 = 50% of avg) */
} ExecNodeDef;

ExecNodeDef EXEC_NODES[] = {
    {2000, "EXEC_ADD",        "Arithmetic: addition",           0.5f},
    {2001, "EXEC_TEXT_GEN",   "Syscall: LLM text generation",   0.6f},
    {2002, "EXEC_TTS",        "Syscall: Text-to-speech",        0.6f},
    {2003, "EXEC_STT",        "Syscall: Speech-to-text",        0.6f},
    {2004, "EXEC_VISION",     "Syscall: Vision identification", 0.6f},
    {2005, "EXEC_CONCAT",     "String: concatenate",            0.5f},
    {2006, "EXEC_SELECT",     "Select: highest activation",     0.5f},
    {2007, "EXEC_WRITE_PORT", "Write: to output port",          0.4f},
    {0, NULL, NULL, 0.0f}  /* Sentinel */
};

/* Semantic routing rules: which patterns route to which EXEC */
typedef struct {
    const char *pattern_type;
    uint32_t exec_node_id;
    float initial_weight;
    const char *example;
} RoutingRule;

RoutingRule ROUTING_RULES[] = {
    /* Arithmetic patterns → EXEC_ADD */
    {"arithmetic", 2000, 0.5f, "2+2, 3+5, etc."},
    
    /* Question patterns → EXEC_TEXT_GEN (LLM) */
    {"question", 2001, 0.6f, "What is X?, How does Y?"},
    
    /* Output/speech patterns → EXEC_TTS */
    {"speak", 2002, 0.7f, "Say X, Speak Y"},
    
    /* Audio input → EXEC_STT */
    {"audio", 2003, 0.6f, "Audio from mic"},
    
    /* Visual patterns → EXEC_VISION */
    {"visual", 2004, 0.6f, "Camera input, image"},
    
    /* Text composition → EXEC_CONCAT */
    {"compose", 2005, 0.5f, "Combine words"},
    
    /* All patterns → EXEC_WRITE_PORT (general output) */
    {"output", 2007, 0.3f, "Any pattern can output"},
    
    {NULL, 0, 0.0f, NULL}  /* Sentinel */
};

/* Create routing edges */
int create_routing_edges(Graph *g) {
    int edges_created = 0;
    
    printf("\n=== Creating Routing Edges ===\n\n");
    
    /* For each EXEC node, create edges from relevant patterns */
    /* Strategy: Create weak edges from ALL patterns initially */
    /* Let graph strengthen the useful ones through feedback */
    
    /* Find all pattern nodes (840+) */
    for (uint64_t pattern_id = 840; pattern_id < g->node_count && pattern_id < 2000; pattern_id++) {
        if (g->nodes[pattern_id].pattern_data_offset > 0) {
            /* This is a pattern node */
            
            /* Create weak edges to multiple EXEC nodes */
            /* Graph will learn which ones are useful */
            
            for (int i = 0; EXEC_NODES[i].node_id != 0; i++) {
                uint32_t exec_id = EXEC_NODES[i].node_id;
                
                /* Check if edge already exists */
                uint32_t eid = g->nodes[pattern_id].first_out;
                int found = 0;
                int checked = 0;
                
                while (eid != UINT32_MAX && eid < g->edge_count && checked++ < 100) {
                    if (g->edges[eid].dst == exec_id) {
                        found = 1;
                        break;
                    }
                    eid = g->edges[eid].next_out;
                }
                
                if (!found) {
                    /* Create weak exploratory edge */
                    float weight = 0.2f;  /* Weak - graph must strengthen if useful */
                    
                    /* Special cases: stronger initial weights */
                    if (exec_id == 2007) {
                        /* EXEC_WRITE_PORT - all patterns should output */
                        weight = 0.4f;
                    }
                    
                    /* Create edge using public API */
                    melvin_create_edge(g, pattern_id, exec_id, weight);
                    edges_created++;
                }
            }
        }
    }
    
    return edges_created;
}

/* Create port-based routing (bypass patterns for direct I/O) */
int create_port_routing(Graph *g) {
    int edges_created = 0;
    
    printf("\n=== Creating Port→EXEC Routing ===\n\n");
    
    /* Input ports → processing EXEC */
    /* Port 0 (mic) → EXEC_STT */
    melvin_create_edge(g, 0, 2003, 0.6f);
    printf("  Port 0 (mic) → EXEC_STT (2003)\n");
    edges_created++;
    
    /* Port 1 (camera) → EXEC_VISION */
    melvin_create_edge(g, 1, 2004, 0.6f);
    printf("  Port 1 (camera) → EXEC_VISION (2004)\n");
    edges_created++;
    
    /* Working memory → EXEC_TEXT_GEN (for questions) */
    for (uint32_t mem = 200; mem < 210; mem++) {
        melvin_create_edge(g, mem, 2001, 0.4f);
    }
    printf("  Memory (200-209) → EXEC_TEXT_GEN (2001)\n");
    edges_created += 10;
    
    /* EXEC outputs → Output ports */
    for (int i = 0; EXEC_NODES[i].node_id != 0; i++) {
        uint32_t exec_id = EXEC_NODES[i].node_id;
        
        /* Route to primary output port (100) */
        melvin_create_edge(g, exec_id, 100, 0.7f);
        
        /* Route to working memory for feedback */
        melvin_create_edge(g, exec_id, 200, 0.5f);
        
        edges_created += 2;
    }
    printf("  All EXEC → Output port (100)\n");
    printf("  All EXEC → Working memory (200)\n");
    
    return edges_created;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <brain.m>\n", argv[0]);
        printf("\nCreates routing edges from patterns to EXEC nodes\n");
        printf("This bootstraps the system for executable intelligence\n");
        return 1;
    }
    
    const char *brain_path = argv[1];
    
    printf("==============================================\n");
    printf("EXEC ROUTING EDGE CREATOR\n");
    printf("==============================================\n\n");
    
    printf("Brain: %s\n", brain_path);
    
    /* Open brain */
    Graph *g = melvin_open(brain_path, 5000, 25000, 8192);
    if (!g) {
        fprintf(stderr, "Failed to open brain (may need to create first)\n");
        return 1;
    }
    
    printf("Loaded: %llu nodes, %llu edges\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    
    /* Count existing patterns */
    int pattern_count = 0;
    for (uint64_t i = 840; i < g->node_count && i < 2000; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_count++;
        }
    }
    
    printf("Patterns found: %d\n", pattern_count);
    
    if (pattern_count == 0) {
        printf("\n⚠ WARNING: No patterns found!\n");
        printf("  Brain needs to learn patterns first (feed it data)\n");
        printf("  Routing edges will be created but won't connect to anything yet.\n\n");
    }
    
    /* Create EXEC nodes (just set them up, code can be added later) */
    printf("\n=== Creating EXEC Nodes ===\n\n");
    
    int exec_created = 0;
    for (int i = 0; EXEC_NODES[i].node_id != 0; i++) {
        uint32_t node_id = EXEC_NODES[i].node_id;
        
        /* For now, point to a placeholder in blob (or 0 to use syscalls) */
        /* Syscall-based EXEC nodes don't need machine code */
        uint64_t blob_offset = 0;  /* 0 = use syscall, not blob code */
        
        if (melvin_create_exec_node(g, node_id, blob_offset, 
                                     EXEC_NODES[i].threshold_ratio) != UINT32_MAX) {
            printf("  ✓ %s (node %u) - %s\n",
                   EXEC_NODES[i].name,
                   node_id,
                   EXEC_NODES[i].description);
            exec_created++;
        }
    }
    
    printf("\nCreated %d EXEC nodes\n", exec_created);
    
    /* Create routing edges */
    int pattern_edges = create_routing_edges(g);
    int port_edges = create_port_routing(g);
    
    int total_edges = pattern_edges + port_edges;
    
    printf("\n==============================================\n");
    printf("SUMMARY\n");
    printf("==============================================\n\n");
    
    printf("Created:\n");
    printf("  EXEC nodes:      %d\n", exec_created);
    printf("  Pattern→EXEC:    %d edges\n", pattern_edges);
    printf("  Port→EXEC:       %d edges\n", port_edges);
    printf("  Total new edges: %d\n\n", total_edges);
    
    printf("Graph now has:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("  Patterns: %d\n", pattern_count);
    printf("  EXEC nodes: %d\n\n", exec_created);
    
    printf("Routing established:\n");
    printf("  • Patterns can route to EXEC nodes (weak edges)\n");
    printf("  • Ports route to appropriate EXEC\n");
    printf("  • EXEC outputs route to ports\n");
    printf("  • Graph will strengthen useful routes through feedback\n\n");
    
    printf("Next steps:\n");
    printf("  1. Feed diverse data (questions, arithmetic, text)\n");
    printf("  2. Let graph learn which routes work\n");
    printf("  3. Edges strengthen/weaken based on success\n");
    printf("  4. System becomes intelligent through usage!\n\n");
    
    /* Sync to disk */
    melvin_sync(g);
    melvin_close(g);
    
    printf("==============================================\n");
    printf("✓ Routing edges created and saved!\n");
    printf("==============================================\n");
    printf("\nBrain ready for: Input → Pattern → EXEC → Output\n");
    
    return 0;
}

