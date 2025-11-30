#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// External helpers (from melvin.c)
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

// MC function: Inspect and report graph state
void mc_inspect_graph(Brain *g, uint64_t node_id) {
    // Only run if activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    printf("\n========================================\n");
    printf("GRAPH INSPECTION REPORT (Tick %llu)\n", (unsigned long long)g->header->tick);
    printf("========================================\n\n");
    
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    printf("BASIC STATS:\n");
    printf("  Total Nodes: %llu\n", (unsigned long long)n);
    printf("  Total Edges: %llu\n", (unsigned long long)e_count);
    printf("\n");
    
    // Count nodes by kind
    uint64_t kind_counts[16] = {0};
    const char *kind_names[] = {
        "DATA", "BLANK", "PATTERN_ROOT", "CONTROL", 
        "TAG", "META", "UNUSED", "UNUSED",
        "UNUSED", "UNUSED", "UNUSED", "UNUSED",
        "UNUSED", "UNUSED", "UNUSED", "UNUSED"
    };
    
    uint64_t mc_nodes = 0;
    uint64_t active_nodes = 0;
    uint64_t pattern_roots = 0;
    uint64_t blank_nodes = 0;
    uint64_t data_nodes = 0;
    uint64_t control_nodes = 0;
    
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &g->nodes[i];
        if (node->kind < 16) kind_counts[node->kind]++;
        
        if (node->kind == NODE_KIND_PATTERN_ROOT) pattern_roots++;
        if (node->kind == NODE_KIND_BLANK) blank_nodes++;
        if (node->kind == NODE_KIND_DATA) data_nodes++;
        if (node->kind == NODE_KIND_CONTROL) control_nodes++;
        
        if (node->mc_id > 0) mc_nodes++;
        if (node->a > 0.1f) active_nodes++;
    }
    
    printf("NODES BY KIND:\n");
    for (int i = 0; i < 16; i++) {
        if (kind_counts[i] > 0) {
            printf("  %s: %llu\n", kind_names[i] ? kind_names[i] : "UNKNOWN", 
                   (unsigned long long)kind_counts[i]);
        }
    }
    printf("\n");
    
    printf("SPECIAL COUNTS:\n");
    printf("  Pattern Roots: %llu\n", (unsigned long long)pattern_roots);
    printf("  Blank Nodes: %llu\n", (unsigned long long)blank_nodes);
    printf("  Data Nodes: %llu\n", (unsigned long long)data_nodes);
    printf("  Control Nodes: %llu\n", (unsigned long long)control_nodes);
    printf("  MC Nodes (have mc_id): %llu\n", (unsigned long long)mc_nodes);
    printf("  Active Nodes (a > 0.1): %llu\n", (unsigned long long)active_nodes);
    printf("\n");
    
    // Count edges by flags
    uint64_t seq_edges = 0;
    uint64_t bind_edges = 0;
    uint64_t control_edges = 0;
    uint64_t pattern_edges = 0;
    uint64_t chan_edges = 0;
    uint64_t active_edges = 0;
    
    for (uint64_t i = 0; i < e_count; i++) {
        Edge *e = &g->edges[i];
        if (e->flags & EDGE_FLAG_SEQ) seq_edges++;
        if (e->flags & EDGE_FLAG_BIND) bind_edges++;
        if (e->flags & EDGE_FLAG_CONTROL) control_edges++;
        if (e->flags & EDGE_FLAG_PATTERN) pattern_edges++;
        if (e->flags & EDGE_FLAG_CHAN) chan_edges++;
        if (fabsf(e->w) > 0.01f) active_edges++;
    }
    
    printf("EDGES BY TYPE:\n");
    printf("  SEQ (sequence): %llu\n", (unsigned long long)seq_edges);
    printf("  BIND (binding): %llu\n", (unsigned long long)bind_edges);
    printf("  CONTROL: %llu\n", (unsigned long long)control_edges);
    printf("  PATTERN: %llu\n", (unsigned long long)pattern_edges);
    printf("  CHAN (channel): %llu\n", (unsigned long long)chan_edges);
    printf("  Active (w != 0): %llu\n", (unsigned long long)active_edges);
    printf("\n");
    
    // Show first 10 pattern roots
    printf("SAMPLE PATTERN ROOTS (first 10):\n");
    uint64_t shown = 0;
    for (uint64_t i = 0; i < n && shown < 10; i++) {
        Node *node = &g->nodes[i];
        if (node->kind == NODE_KIND_PATTERN_ROOT) {
            printf("  Node %llu: a=%.3f bias=%.3f value=%.0f\n", 
                   (unsigned long long)i, node->a, node->bias, node->value);
            shown++;
        }
    }
    if (pattern_roots > 10) {
        printf("  ... and %llu more\n", (unsigned long long)(pattern_roots - 10));
    }
    printf("\n");
    
    // Show first 10 blank nodes
    printf("SAMPLE BLANK NODES (first 10):\n");
    shown = 0;
    for (uint64_t i = 0; i < n && shown < 10; i++) {
        Node *node = &g->nodes[i];
        if (node->kind == NODE_KIND_BLANK) {
            printf("  Node %llu: a=%.3f value=%.0f\n", 
                   (unsigned long long)i, node->a, node->value);
            shown++;
        }
    }
    if (blank_nodes > 10) {
        printf("  ... and %llu more\n", (unsigned long long)(blank_nodes - 10));
    }
    printf("\n");
    
    // Show first 10 edges from pattern roots
    printf("SAMPLE EDGES FROM PATTERN ROOTS (first 10):\n");
    shown = 0;
    for (uint64_t pid = 0; pid < n && shown < 10; pid++) {
        if (g->nodes[pid].kind == NODE_KIND_PATTERN_ROOT) {
            for (uint64_t eid = 0; eid < e_count && shown < 10; eid++) {
                Edge *e = &g->edges[eid];
                if (e->src == pid && (e->flags & EDGE_FLAG_PATTERN)) {
                    const char *type = "?";
                    if (e->flags & EDGE_FLAG_BIND) type = "BIND";
                    else if (e->flags & EDGE_FLAG_CHAN) type = "CHAN";
                    
                    printf("  Pattern %llu -> Node %llu (w=%.2f, %s)\n",
                           (unsigned long long)pid, (unsigned long long)e->dst, 
                           e->w, type);
                    shown++;
                    break; // Only show one edge per pattern
                }
            }
        }
    }
    printf("\n");
    
    // Show MC nodes
    printf("MC NODES (nodes with mc_id > 0):\n");
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &g->nodes[i];
        if (node->mc_id > 0) {
            const char *kind_str = kind_names[node->kind] ? kind_names[node->kind] : "?";
            printf("  Node %llu: kind=%s mc_id=%u a=%.3f bias=%.3f\n",
                   (unsigned long long)i, kind_str, node->mc_id, node->a, node->bias);
        }
    }
    printf("\n");
    
    printf("========================================\n");
    printf("END INSPECTION REPORT\n");
    printf("========================================\n\n");
    
    // Deactivate after running once
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

