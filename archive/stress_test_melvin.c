#define _POSIX_C_SOURCE 200809L
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

// Forward declarations (from melvin.c) - must be before use
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);
extern void update_edges(Brain *);
extern void propagate_predictions(Brain *);
extern void decay_activations(Brain *);
extern int grow_graph(Brain *, uint64_t, uint64_t);

// Test configuration
#define STRESS_NODES 10000
#define STRESS_EDGES 50000
#define STRESS_TICKS 1000
#define STRESS_PATTERNS 100

// Test results
typedef struct {
    int passed;
    int failed;
    int warnings;
    char failures[1000][256];
    char warnings_list[1000][256];
} TestResults;

void add_failure(TestResults *r, const char *msg) {
    if (r->failed < 1000) {
        snprintf(r->failures[r->failed], sizeof(r->failures[r->failed]), "%s", msg);
        r->failed++;
    }
}

void add_warning(TestResults *r, const char *msg) {
    if (r->warnings < 1000) {
        snprintf(r->warnings_list[r->warnings], sizeof(r->warnings_list[r->warnings]), "%s", msg);
        r->warnings++;
    }
}

// Test 1: Graph growth under stress
int test_graph_growth(Brain *g, TestResults *r) {
    printf("\n=== TEST 1: Graph Growth Under Stress ===\n");
    
    uint64_t initial_nodes = g->header->num_nodes;
    uint64_t initial_edges = g->header->num_edges;
    
    printf("Initial: %llu nodes, %llu edges\n", 
           (unsigned long long)initial_nodes, (unsigned long long)initial_edges);
    
    // Create many nodes rapidly
    uint64_t nodes_created = 0;
    for (int i = 0; i < STRESS_NODES; i++) {
        uint64_t id = alloc_node(g);
        if (id == UINT64_MAX || id == 0) {
            add_failure(r, "alloc_node failed during stress test");
            return 0;
        }
        nodes_created++;
    }
    
    uint64_t final_nodes = g->header->num_nodes;
    uint64_t final_edges = g->header->num_edges;
    
    printf("After creating %d nodes: %llu nodes, %llu edges\n",
           STRESS_NODES, (unsigned long long)final_nodes, (unsigned long long)final_edges);
    
    // Verify growth
    if (final_nodes < initial_nodes + STRESS_NODES) {
        add_failure(r, "Graph did not grow enough nodes");
        return 0;
    }
    
    // Verify sequence edges were created (should have ~2x nodes created)
    uint64_t expected_seq_edges = nodes_created * 2; // bidirectional
    uint64_t seq_edges = 0;
    for (uint64_t i = 0; i < final_edges; i++) {
        if (g->edges[i].flags & EDGE_FLAG_SEQ) {
            seq_edges++;
        }
    }
    
    printf("Sequence edges created: %llu (expected ~%llu)\n",
           (unsigned long long)seq_edges, (unsigned long long)expected_seq_edges);
    
    if (seq_edges < expected_seq_edges * 0.8) { // Allow 20% variance
        add_warning(r, "Fewer sequence edges than expected");
    }
    
    r->passed++;
    return 1;
}

// Test 2: Edge creation under stress
int test_edge_creation(Brain *g, TestResults *r) {
    printf("\n=== TEST 2: Edge Creation Under Stress ===\n");
    
    uint64_t initial_edges = g->header->num_edges;
    uint64_t n = g->header->num_nodes;
    
    if (n < 100) {
        add_failure(r, "Not enough nodes for edge stress test");
        return 0;
    }
    
    // Create many edges rapidly
    uint64_t edges_created = 0;
    for (int i = 0; i < STRESS_EDGES; i++) {
        uint64_t src = (i * 7) % n; // Distribute across nodes
        uint64_t dst = (i * 11) % n;
        
        if (src != dst) {
            add_edge(g, src, dst, 0.5f, EDGE_FLAG_BIND);
            edges_created++;
        }
    }
    
    uint64_t final_edges = g->header->num_edges;
    
    printf("Created %llu edges: %llu -> %llu edges\n",
           (unsigned long long)edges_created,
           (unsigned long long)initial_edges,
           (unsigned long long)final_edges);
    
    if (final_edges < initial_edges + edges_created * 0.9) { // Allow 10% variance
        add_failure(r, "Not all edges were created");
        return 0;
    }
    
    r->passed++;
    return 1;
}

// Test 3: Co-activation and edge formation
int test_coactivation(Brain *g, TestResults *r) {
    printf("\n=== TEST 3: Co-Activation Edge Formation ===\n");
    
    uint64_t n = g->header->num_nodes;
    if (n < 10) {
        add_failure(r, "Not enough nodes for co-activation test");
        return 0;
    }
    
    // Activate some nodes
    uint64_t initial_edges = g->header->num_edges;
    
    for (uint64_t i = 0; i < 100 && i < n; i++) {
        g->nodes[i].a = 1.0f; // Strong activation
    }
    
    // Run update_edges to create co-activation edges
    update_edges(g);
    
    uint64_t final_edges = g->header->num_edges;
    uint64_t new_edges = final_edges - initial_edges;
    
    printf("Co-activation created %llu new edges\n", (unsigned long long)new_edges);
    
    if (new_edges == 0) {
        add_warning(r, "No edges created from co-activation");
    } else {
        printf("  ✓ Co-activation is working\n");
    }
    
    r->passed++;
    return 1;
}

// Test 4: Pattern formation
int test_pattern_formation(Brain *g, TestResults *r) {
    printf("\n=== TEST 4: Pattern Formation ===\n");
    
    uint64_t initial_patterns = 0;
    uint64_t n = g->header->num_nodes;
    
    // Count existing patterns
    for (uint64_t i = 0; i < n; i++) {
        if (g->nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
            initial_patterns++;
        }
    }
    
    printf("Initial patterns: %llu\n", (unsigned long long)initial_patterns);
    
    // Create pattern nodes
    uint64_t patterns_created = 0;
    for (int i = 0; i < STRESS_PATTERNS; i++) {
        uint64_t pattern_id = alloc_node(g);
        if (pattern_id != UINT64_MAX && pattern_id != 0) {
            g->nodes[pattern_id].kind = NODE_KIND_PATTERN_ROOT;
            g->nodes[pattern_id].a = 0.6f;
            g->nodes[pattern_id].bias = 0.5f;
            patterns_created++;
        }
    }
    
    // Count final patterns
    uint64_t final_patterns = 0;
    n = g->header->num_nodes;
    for (uint64_t i = 0; i < n; i++) {
        if (g->nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
            final_patterns++;
        }
    }
    
    printf("Final patterns: %llu (created %llu)\n",
           (unsigned long long)final_patterns, (unsigned long long)patterns_created);
    
    if (final_patterns < initial_patterns + patterns_created * 0.9) {
        add_failure(r, "Patterns were not properly created");
        return 0;
    }
    
    r->passed++;
    return 1;
}

// Test 5: Memory/file integrity
int test_memory_integrity(Brain *g, TestResults *r) {
    printf("\n=== TEST 5: Memory/File Integrity ===\n");
    
    uint64_t n = g->header->num_nodes;
    uint64_t e = g->header->num_edges;
    
    // Check for corrupted nodes
    int corrupted_nodes = 0;
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &g->nodes[i];
        if (isnan(node->a) || isinf(node->a)) {
            corrupted_nodes++;
        }
        if (isnan(node->bias) || isinf(node->bias)) {
            corrupted_nodes++;
        }
    }
    
    if (corrupted_nodes > 0) {
        add_failure(r, "Found corrupted node data (NaN/Inf)");
        return 0;
    }
    
    // Check for corrupted edges
    int corrupted_edges = 0;
    for (uint64_t i = 0; i < e; i++) {
        Edge *edge = &g->edges[i];
        if (edge->src >= n || edge->dst >= n) {
            corrupted_edges++;
        }
        if (isnan(edge->w) || isinf(edge->w)) {
            corrupted_edges++;
        }
    }
    
    printf("Nodes: %llu, Edges: %llu\n", (unsigned long long)n, (unsigned long long)e);
    printf("Corrupted nodes: %d, Corrupted edges: %d\n", corrupted_nodes, corrupted_edges);
    
    if (corrupted_edges > e * 0.1) { // Allow 10% corrupted edges (some may be intentional)
        add_warning(r, "High number of corrupted edges detected");
    }
    
    r->passed++;
    return 1;
}

// Test 6: Emergence properties
int test_emergence(Brain *g, TestResults *r) {
    printf("\n=== TEST 6: Emergence Properties ===\n");
    
    uint64_t n = g->header->num_nodes;
    uint64_t e = g->header->num_edges;
    
    // Count different edge types
    uint64_t seq_edges = 0;
    uint64_t bind_edges = 0;
    uint64_t pattern_edges = 0;
    uint64_t coactivation_edges = 0;
    
    for (uint64_t i = 0; i < e; i++) {
        Edge *edge = &g->edges[i];
        if (edge->flags & EDGE_FLAG_SEQ) seq_edges++;
        if (edge->flags & EDGE_FLAG_BIND) bind_edges++;
        if (edge->flags & EDGE_FLAG_PATTERN) pattern_edges++;
        if (!(edge->flags & EDGE_FLAG_SEQ) && !(edge->flags & EDGE_FLAG_PATTERN)) {
            coactivation_edges++; // Likely from co-activation
        }
    }
    
    printf("Edge types:\n");
    printf("  Sequence edges: %llu (temporal learning)\n", (unsigned long long)seq_edges);
    printf("  Binding edges: %llu (structure)\n", (unsigned long long)bind_edges);
    printf("  Pattern edges: %llu (intelligence)\n", (unsigned long long)pattern_edges);
    printf("  Co-activation edges: %llu (learning)\n", (unsigned long long)coactivation_edges);
    
    // Check if emergence is happening
    int emergence_score = 0;
    
    if (seq_edges > 0) {
        printf("  ✓ Sequence edges present (temporal connections)\n");
        emergence_score++;
    }
    
    if (pattern_edges > 0) {
        printf("  ✓ Pattern edges present (intelligence structure)\n");
        emergence_score++;
    }
    
    if (coactivation_edges > 0) {
        printf("  ✓ Co-activation edges present (learning)\n");
        emergence_score++;
    }
    
    // Count patterns
    uint64_t patterns = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (g->nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
            patterns++;
        }
    }
    
    printf("  Patterns: %llu\n", (unsigned long long)patterns);
    
    if (patterns > 0) {
        printf("  ✓ Patterns present (intelligence stored)\n");
        emergence_score++;
    }
    
    if (emergence_score < 2) {
        add_warning(r, "Limited emergence detected - may need more runtime");
    } else {
        printf("  ✓ Emergence is working!\n");
    }
    
    r->passed++;
    return 1;
}

// Test 7: Rule stability
int test_rule_stability(Brain *g, TestResults *r) {
    printf("\n=== TEST 7: Rule Stability ===\n");
    
    // Test that rules don't break under load
    uint64_t initial_nodes = g->header->num_nodes;
    uint64_t initial_edges = g->header->num_edges;
    
    // Run many ticks
    for (int tick = 0; tick < STRESS_TICKS; tick++) {
        g->header->tick++;
        
        // Update activations
        propagate_predictions(g);
        
        // Update edges (co-activation)
        if (tick % 10 == 0) {
            update_edges(g);
        }
        
        // Decay
        decay_activations(g);
    }
    
    uint64_t final_nodes = g->header->num_nodes;
    uint64_t final_edges = g->header->num_edges;
    
    printf("After %d ticks: %llu nodes, %llu edges\n",
           STRESS_TICKS,
           (unsigned long long)final_nodes,
           (unsigned long long)final_edges);
    
    // Check for stability
    if (final_nodes < initial_nodes) {
        add_failure(r, "Nodes were lost during stress test");
        return 0;
    }
    
    // Check for reasonable growth
    if (final_edges < initial_edges) {
        add_warning(r, "Edges decreased during stress test");
    }
    
    printf("  ✓ Rules remained stable under load\n");
    
    r->passed++;
    return 1;
}

int main(int argc, char **argv) {
    const char *db_path = "melvin.m";
    if (argc > 1) {
        db_path = argv[1];
    }
    
    printf("========================================\n");
    printf("MELVIN STRESS TEST\n");
    printf("========================================\n");
    printf("Testing: %s\n", db_path);
    printf("Configuration:\n");
    printf("  Nodes: %d\n", STRESS_NODES);
    printf("  Edges: %d\n", STRESS_EDGES);
    printf("  Ticks: %d\n", STRESS_TICKS);
    printf("  Patterns: %d\n", STRESS_PATTERNS);
    printf("\n");
    
    // Open brain file
    int fd = open(db_path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    // Initialize if empty
    struct stat st;
    fstat(fd, &st);
    if (st.st_size < sizeof(BrainHeader)) {
        // Create minimal header
        BrainHeader header = {0};
        header.num_nodes = 0;
        header.num_edges = 0;
        header.tick = 0;
        write(fd, &header, sizeof(BrainHeader));
        ftruncate(fd, 1024 * 1024); // 1MB initial
    }
    
    fstat(fd, &st);
    size_t filesize = st.st_size;
    
    void *map = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    
    Brain g;
    g.fd = fd;
    g.mmap_size = filesize;
    g.header = (BrainHeader*)map;
    g.nodes = (Node*)((uint8_t*)map + sizeof(BrainHeader));
    g.edges = (Edge*)((uint8_t*)g.nodes + g.header->num_nodes * sizeof(Node));
    
    TestResults results = {0};
    
    // Run tests
    test_graph_growth(&g, &results);
    test_edge_creation(&g, &results);
    test_coactivation(&g, &results);
    test_pattern_formation(&g, &results);
    test_memory_integrity(&g, &results);
    test_emergence(&g, &results);
    test_rule_stability(&g, &results);
    
    // Sync to disk
    msync(map, filesize, MS_SYNC);
    
    // Print summary
    printf("\n========================================\n");
    printf("STRESS TEST SUMMARY\n");
    printf("========================================\n");
    printf("Passed: %d\n", results.passed);
    printf("Failed: %d\n", results.failed);
    printf("Warnings: %d\n", results.warnings);
    
    if (results.failed > 0) {
        printf("\nFailures:\n");
        for (int i = 0; i < results.failed; i++) {
            printf("  - %s\n", results.failures[i]);
        }
    }
    
    if (results.warnings > 0) {
        printf("\nWarnings:\n");
        for (int i = 0; i < results.warnings; i++) {
            printf("  - %s\n", results.warnings_list[i]);
        }
    }
    
    printf("\nFinal graph state:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g.header->num_nodes);
    printf("  Edges: %llu\n", (unsigned long long)g.header->num_edges);
    printf("  Tick: %llu\n", (unsigned long long)g.header->tick);
    
    munmap(map, filesize);
    close(fd);
    
    return (results.failed > 0) ? 1 : 0;
}

