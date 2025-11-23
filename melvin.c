#define _POSIX_C_SOURCE 200809L
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <time.h>
#include <dlfcn.h> // Added for dynamic loading
#include <math.h>
#include <inttypes.h>
#include <stdint.h>

// Debug flag
static int g_debug = 0;
static int g_log_simplicity = 0; // Enable via env var MELVIN_LOG_SIMPLICITY=1

// ========================================================
// Simplicity Objective Metrics
// ========================================================

// Channel types for prediction error tracking
#define CH_TEXT    1
#define CH_SENSOR  2
#define CH_MOTOR   3
#define CH_VISION  4
#define CH_REWARD  4  // Reuse same value if no separate reward channel yet

typedef struct {
    // Prediction error (how badly we failed to predict inputs)
    double pred_error_total;
    double pred_error_text;
    double pred_error_sensor;
    double pred_error_motor;
    
    // Size / complexity
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t num_patterns;
    
    // Compression / reuse proxies
    double avg_pattern_length;      // avg number of slots per pattern (approx)
    double pattern_usage_rate;      // fraction of active nodes that belong to patterns
    double episodic_compression;    // ratio: raw bytes vs pattern-explained bytes (approx)
    
    // Derived objective
    double simplicity_score;        // higher is "better" (more intelligent / simpler)
} SimplicityMetrics;

// Global metrics accumulator (reset each tick)
static SimplicityMetrics g_simplicity_metrics = {0};

// ========================================================
// MC Table Definition
// ========================================================

#define MAX_MC_FUNCS 256

// Forward declaration not needed if melvin.h is included and defines Brain
// struct Brain;
// typedef struct Brain Brain;

typedef void (*MCFn)(Brain *g, uint64_t node_id);

typedef struct {
    const char *name;
    MCFn        fn;
    uint32_t    flags;
} MCEntry;

MCEntry  g_mc_table[MAX_MC_FUNCS];  // Made non-static for bootstrap access
uint32_t g_mc_count = 0;

// ========================================================
// Helper Functions
// ========================================================

// Find a free node (exported for plugins)
uint64_t alloc_node(Brain *g) {
    if (g->header->num_nodes >= g->header->node_cap) return 0; // 0 is usually reserved or valid, but let's check
    // Assume append for now
    uint64_t id = g->header->num_nodes++;
    memset(&g->nodes[id], 0, sizeof(Node));
    return id;
}

// Add edge (exported for plugins)
void add_edge(Brain *g, uint64_t src, uint64_t dst, float w, uint32_t flags) {
    if (g->header->num_edges >= g->header->edge_cap) return;
    
    // Simple linear scan to update existing edge? 
    // For speed, let's just append. Compaction is for later.
    // Or check if exists? Expensive for O(E).
    // Just append new edges.
    
    Edge *e = &g->edges[g->header->num_edges++];
    e->src = src;
    e->dst = dst;
    e->w = w;
    e->flags = flags;
    e->usage_count = 1;
}

// ========================================================
// MC Functions - All moved to plugins
// ========================================================
// All MC functions are now in plugins/ and loaded dynamically

// ========================================================
// Core Runtime
// ========================================================

// Helper to compile and load a plugin
static MCFn load_plugin_function(const char *plugin_name, const char *func_name) {
    char so_path[256];
    char src_path[256];
    char cmd[512];
    
    snprintf(so_path, sizeof(so_path), "plugins/%s.so", plugin_name);
    snprintf(src_path, sizeof(src_path), "plugins/%s.c", plugin_name);
    
    // Compile if needed
    if (access(so_path, F_OK) != 0) {
        fprintf(stderr, "[main] Compiling %s...\n", src_path);
        snprintf(cmd, sizeof(cmd), "clang -shared -fPIC -O2 -I. -undefined dynamic_lookup -o %s %s", so_path, src_path);
        int rc = system(cmd);
        if (rc != 0) {
            fprintf(stderr, "[main] Failed to compile %s\n", src_path);
            return NULL;
        }
    }
    
    // Load plugin
    void *h = dlopen(so_path, RTLD_NOW);
    if (!h) {
        fprintf(stderr, "[main] Failed to load %s: %s\n", so_path, dlerror());
        return NULL;
    }
    
    MCFn fn = (MCFn)dlsym(h, func_name);
    if (!fn) {
        fprintf(stderr, "[main] Failed to find %s in %s: %s\n", func_name, so_path, dlerror());
        dlclose(h);
        return NULL;
    }
    
    return fn;
}

void register_mc(const char *name, MCFn fn) {
    if (g_mc_count >= MAX_MC_FUNCS) return;
    g_mc_table[g_mc_count].name = name;
    g_mc_table[g_mc_count].fn = fn;
    g_mc_table[g_mc_count].flags = 0;
    g_mc_count++;
}

void run_mc_nodes(Brain *g) {
    for (uint64_t i = 0; i < g->header->num_nodes; ++i) {
        Node *n = &g->nodes[i];
        if (n->mc_id == 0) continue;
        if (n->a < 0.5f) continue;
        if (n->mc_id < MAX_MC_FUNCS) {
            MCEntry *entry = &g_mc_table[n->mc_id];
            if (entry->fn) {
                entry->fn(g, i);
            } else {
                fprintf(stderr, "MC function missing for id %u\n", n->mc_id);
            }
        }
    }
}

#include <math.h>

// ... (previous includes)

// Global transient buffers
static float *g_predicted_a = NULL;
static float *g_node_error = NULL;
static uint64_t g_buffer_cap = 0;

static void ensure_buffers(Brain *g) {
    if (g->header->node_cap > g_buffer_cap) {
        g_buffer_cap = g->header->node_cap;
        g_predicted_a = realloc(g_predicted_a, g_buffer_cap * sizeof(float));
        g_node_error = realloc(g_node_error, g_buffer_cap * sizeof(float));
        // Zero new parts if necessary, but we usually overwrite
    }
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void ingest_input(Brain *g) {
    // Input ingestion handled by MC nodes or external systems
    // This is a placeholder for any runtime-level input processing
}

void propagate_predictions(Brain *g) {
    ensure_buffers(g);
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;

    // 1. Reset predictions to bias
    for(uint64_t i=0; i<n; i++) {
        g_predicted_a[i] = g->nodes[i].bias;
    }

    // 2. Sum weighted inputs from edges
    for(uint64_t i=0; i<e_count; i++) {
        Edge *e = &g->edges[i];
        if (e->src < n && e->dst < n) {
            float input = e->w * g->nodes[e->src].a;
            g_predicted_a[e->dst] += input;
        }
    }

    // 3. Apply nonlinearity and clamp to [0, 1]
    for(uint64_t i=0; i<n; i++) {
        g_predicted_a[i] = sigmoid(g_predicted_a[i]);
        // Clamp to prevent explosion
        if (g_predicted_a[i] < 0.0f) g_predicted_a[i] = 0.0f;
        if (g_predicted_a[i] > 1.0f) g_predicted_a[i] = 1.0f;
    }
    
    // Note: We do NOT modify actual activations here.
    // Actual activations are set by ingest_input, MC nodes, or apply_environment.
}

void apply_environment(Brain *g) {
    // Apply decay and normalization
    uint64_t n = g->header->num_nodes;
    for(uint64_t i=0; i<n; i++) {
        Node *node = &g->nodes[i];
        // Decay activation
        node->a *= (1.0f - node->decay);
        // Clamp to [0, 1]
        if (node->a < 0.0f) node->a = 0.0f;
        if (node->a > 1.0f) node->a = 1.0f;
    }
}

// Accumulate prediction error into simplicity metrics
static void sm_accumulate_prediction_error(SimplicityMetrics *m, double err, int channel_type) {
    double err_abs = fabs(err);
    m->pred_error_total += err_abs;
    switch (channel_type) {
        case CH_TEXT:   m->pred_error_text   += err_abs; break;
        case CH_SENSOR: m->pred_error_sensor += err_abs; break;
        case CH_MOTOR:  m->pred_error_motor  += err_abs; break;
        default: break;
    }
}

void compute_error(Brain *g) {
    ensure_buffers(g);
    uint64_t n = g->header->num_nodes;
    
    // Reset error accumulation for this tick
    double total_error = 0.0;
    
    for(uint64_t i=0; i<n; i++) {
        float a_actual = g->nodes[i].a;
        float a_pred = g_predicted_a[i];
        float e = a_actual - a_pred;
        // Clamp error to reasonable range
        if (e > 1.0f) e = 1.0f;
        if (e < -1.0f) e = -1.0f;
        g_node_error[i] = e;
        
        // Accumulate into simplicity metrics
        // Assume all nodes contribute to general prediction error
        // Channel-specific tracking would require metadata on nodes
        total_error += fabs(e);
    }
    
    // Accumulate total prediction error (approximate - treating all nodes as general input)
    sm_accumulate_prediction_error(&g_simplicity_metrics, total_error / (double)n, 0);
}

void update_edges(Brain *g) {
    uint64_t e_count = g->header->num_edges;
    const float eta = 0.001f;  // Learning rate
    const float W_MAX = 10.0f;
    const float lambda = 0.9f; // Eligibility trace decay

    for(uint64_t i=0; i<e_count; i++) {
        Edge *e = &g->edges[i];
        if (e->src >= g->header->num_nodes || e->dst >= g->header->num_nodes) continue;
        
        Node *src = &g->nodes[e->src];
        float a_src = src->a;
        float err_dst = g_node_error[e->dst];
        
        // Simple local learning rule: Δw = η * a_src * err_dst
        float dw = eta * a_src * err_dst;
        
        // NaN guard (check for NaN or infinity)
        if (dw != dw || dw > 1e10f || dw < -1e10f) dw = 0.0f;
        
        e->w += dw;
        
        // Clamp weights
        if (e->w > W_MAX) e->w = W_MAX;
        if (e->w < -W_MAX) e->w = -W_MAX;
        
        // Update eligibility trace
        e->elig = lambda * e->elig + a_src;
        
        // Update usage count when edge contributes
        if (fabsf(a_src) > 0.1f) {
            e->usage_count++;
        }
    }
}

void update_nodes_from_error(Brain *g) {
    uint64_t n = g->header->num_nodes;
    
    for(uint64_t i=0; i<n; i++) {
        Node *node = &g->nodes[i];
        float err_abs = fabsf(g_node_error[i]);
        
        // Update reliability: nodes with low error get high reliability
        float reliability_update = 1.0f - fminf(1.0f, err_abs);
        node->reliability = 0.99f * node->reliability + 0.01f * reliability_update;
        
        // Clamp reliability to [0, 1]
        if (node->reliability < 0.0f) node->reliability = 0.0f;
        if (node->reliability > 1.0f) node->reliability = 1.0f;
        
        // Track success/failure counts
        if (err_abs < 0.1f) {
            node->success_count++;
                } else {
            node->failure_count++;
        }
    }
}


void log_learning_stats(Brain *g) {
    if (!g_debug) return;
    if (g->header->tick % 1000 != 0) return;
    
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    // Count edges with non-zero weight
    uint64_t active_edges = 0;
    for(uint64_t i=0; i<e_count; i++) {
        if (fabsf(g->edges[i].w) > 0.01f) active_edges++;
    }
    
    // Count nodes by kind (generic statistics only)
    uint64_t kind_counts[16] = {0};
    uint64_t mc_nodes = 0;
    uint64_t active_nodes = 0;
    
    for(uint64_t i=0; i<n; i++) {
        Node *node = &g->nodes[i];
        if (node->kind < 16) kind_counts[node->kind]++;
        if (node->mc_id > 0) mc_nodes++;
        if (node->a > 0.1f) active_nodes++;
    }
    
    fprintf(stderr, "[tick %llu] nodes=%llu edges=%llu active_edges=%llu active_nodes=%llu mc_nodes=%llu",
            (unsigned long long)g->header->tick,
            (unsigned long long)n,
            (unsigned long long)e_count,
            (unsigned long long)active_edges,
            (unsigned long long)active_nodes,
            (unsigned long long)mc_nodes);
    
    fprintf(stderr, "\n");
}

// ========================================================
// Simplicity Metrics Computation
// ========================================================

// Initialize metrics for a new tick
static void sm_init(SimplicityMetrics *m) {
    memset(m, 0, sizeof(*m));
}

// Measure graph complexity
static void sm_measure_complexity(Brain *g, SimplicityMetrics *m) {
    m->num_nodes = g->header->num_nodes;
    m->num_edges = g->header->num_edges;
    
    // Count patterns (PATTERN_ROOT nodes)
    m->num_patterns = 0;
    uint64_t n = g->header->num_nodes;
    uint64_t total_pattern_nodes = 0; // Nodes connected to patterns
    uint64_t total_pattern_edges = 0; // Edges tagged as PATTERN
    
    for (uint64_t i = 0; i < n; i++) {
        if (g->nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
            m->num_patterns++;
        }
    }
    
    // Count edges with PATTERN flag to approximate pattern coverage
    uint64_t e_count = g->header->num_edges;
    for (uint64_t i = 0; i < e_count; i++) {
        if (g->edges[i].flags & EDGE_FLAG_PATTERN) {
            total_pattern_edges++;
        }
    }
    
    // Approximate nodes in patterns from pattern edges
    // Each pattern edge connects a pattern to a node, so count unique destinations
    uint64_t pattern_connected_nodes = 0;
    for (uint64_t i = 0; i < e_count; i++) {
        if (g->edges[i].flags & EDGE_FLAG_PATTERN && g->edges[i].dst < n) {
            // Count unique nodes (approximate - will double count)
            pattern_connected_nodes++;
        }
    }
    
    m->pattern_usage_rate = (n > 0) ? (double)pattern_connected_nodes / (double)n : 0.0;
    if (m->pattern_usage_rate > 1.0) m->pattern_usage_rate = 1.0; // Clamp
}

// Measure pattern compression/reuse
static void sm_measure_patterns(Brain *g, SimplicityMetrics *m) {
    if (m->num_patterns == 0) {
        m->avg_pattern_length = 0.0;
        m->episodic_compression = 0.0;
        return;
    }
    
    // Count pattern edges per pattern (approximate average length)
    uint64_t e_count = g->header->num_edges;
    uint64_t total_pattern_slots = 0;
    
    // For each pattern root, count outgoing pattern edges
    uint64_t n = g->header->num_nodes;
    for (uint64_t pid = 0; pid < n; pid++) {
        if (g->nodes[pid].kind == NODE_KIND_PATTERN_ROOT) {
            uint64_t slots = 0;
            for (uint64_t i = 0; i < e_count; i++) {
                if (g->edges[i].src == pid && (g->edges[i].flags & EDGE_FLAG_PATTERN)) {
                    slots++;
                }
            }
            total_pattern_slots += slots;
        }
    }
    
    m->avg_pattern_length = (m->num_patterns > 0) ?
        (double)total_pattern_slots / (double)m->num_patterns : 0.0;
    
    // Episodic compression: use pattern_usage_rate as proxy
    // Higher pattern_usage_rate means more data is explained by patterns (compression)
    m->episodic_compression = m->pattern_usage_rate;
}

// Compute simplicity objective score
static void sm_compute_objective(SimplicityMetrics *m) {
    // Hyperparameters: tunable constants
    const double W_PRED   = -1.0;   // penalize prediction error
    const double W_SIZE   = -1e-6;  // small penalty per node/edge
    const double W_COMP   =  1.0;   // reward compression/reuse
    
    double size_penalty = (double)m->num_nodes + (double)m->num_edges;
    
    double score = 0.0;
    score += W_PRED * m->pred_error_total;
    score += W_SIZE * size_penalty;
    score += W_COMP * m->episodic_compression;
    
    m->simplicity_score = score;
}

// Normalize score to reward signal
static float sm_reward_from_score(const SimplicityMetrics *m) {
    double r = m->simplicity_score;
    
    // Normalize to reasonable range for reward signal
    // Scale down to small values that can be used as reward
    r = r * 0.01; // Scale factor
    
    // Clamp to reasonable range
    if (r > 10.0) r = 10.0;
    if (r < -10.0) r = -10.0;
    
    return (float)r;
}

// Inject intrinsic reward into graph
static void melvin_send_intrinsic_reward(Brain *g, float reward_value) {
    // Find or create reward channel node
    // Look for existing META node with reward value
    uint64_t n = g->header->num_nodes;
    uint64_t reward_node = UINT64_MAX;
    
    // Try to find existing reward node (META with specific value)
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &g->nodes[i];
        if (node->kind == NODE_KIND_META && (uint32_t)node->value == 0x52455744) { // "REWD"
            reward_node = i;
            break;
        }
    }
    
    // Create reward node if not found
    if (reward_node == UINT64_MAX) {
        reward_node = alloc_node(g);
        if (reward_node != UINT64_MAX && reward_node < g->header->node_cap) {
            Node *rn = &g->nodes[reward_node];
            rn->kind = NODE_KIND_META;
            rn->value = 0x52455744; // "REWD"
            rn->a = 0.0f;
            rn->bias = 0.0f;
        } else {
            return; // Failed to allocate
        }
    }
    
    // Inject reward as activation spike
    // This reward signal will propagate through the graph and modulate learning
    Node *reward = &g->nodes[reward_node];
    reward->a += reward_value * 0.1f; // Accumulate reward (small scale)
    
    // Clamp activation
    if (reward->a > 1.0f) reward->a = 1.0f;
    if (reward->a < -1.0f) reward->a = -1.0f;
    
    // Also store in value field for direct access
    reward->value = reward_value;
}

// Log simplicity metrics
static void sm_log(const SimplicityMetrics *m, uint64_t tick) {
    if (!g_log_simplicity) return;
    
    fprintf(stderr,
        "[simplicity] tick=%" PRIu64 " score=%.4f pred_total=%.4f size=(nodes=%" PRIu64 ",edges=%" PRIu64 ",patterns=%" PRIu64 ") comp=%.4f usage=%.4f\n",
        tick,
        m->simplicity_score,
        m->pred_error_total,
        m->num_nodes,
        m->num_edges,
        m->num_patterns,
        m->episodic_compression,
        m->pattern_usage_rate);
}

void emit_output(Brain *g) {
    // Stub
}

void melvin_tick(Brain *g) {
    // Initialize simplicity metrics for this tick
    sm_init(&g_simplicity_metrics);
    
    // 0. External input / MC effects
    ingest_input(g);
    
    // 1. Propagate predictions: compute Â
    propagate_predictions(g);
    
    // 2. Apply environment / finalize actual activations (decay, normalization)
    apply_environment(g);
    
    // 3. Compute local errors e_j = A_j - Â_j (this also accumulates into metrics)
    compute_error(g);
    
    // 4. Update weights & node reliability
    update_edges(g);
    update_nodes_from_error(g);
    
    // 5. Run MC-backed nodes chosen by the graph
    run_mc_nodes(g);
    
    // 6. Compute simplicity metrics
    sm_measure_complexity(g, &g_simplicity_metrics);
    sm_measure_patterns(g, &g_simplicity_metrics);
    sm_compute_objective(&g_simplicity_metrics);
    
    // 7. Inject intrinsic reward into graph
    float intrinsic_reward = sm_reward_from_score(&g_simplicity_metrics);
    melvin_send_intrinsic_reward(g, intrinsic_reward);
    
    // 8. Emit outputs if any
    emit_output(g);
    
    // 9. Debug logging
    log_learning_stats(g);
    sm_log(&g_simplicity_metrics, g->header->tick);
    
    g->header->tick++;
}

int main(int argc, char **argv) {
    const char *db_path = "melvin.m";
    if (argc > 1) {
        if (strcmp(argv[1], "-d") == 0 || strcmp(argv[1], "--debug") == 0) {
            g_debug = 1;
            if (argc > 2) db_path = argv[2];
        } else {
            db_path = argv[1];
        }
    }
    
    // Check for simplicity logging flag
    if (getenv("MELVIN_LOG_SIMPLICITY")) {
        g_log_simplicity = 1;
    }
    
    // Initialize simplicity metrics
    sm_init(&g_simplicity_metrics);

    int fd = open(db_path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not open %s. Run melvin_minit first.\n", db_path);
        return 1;
    }

    struct stat st;
    fstat(fd, &st);
    size_t filesize = st.st_size;

    void *map = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    Brain g;
    g.fd = fd;
    g.mmap_size = filesize;
    g.header = (BrainHeader*)map;
    g.nodes = (Node*)((uint8_t*)map + sizeof(BrainHeader));
    // Edges follow nodes. Note: num_nodes in header tracks *used*, but we need *capacity* to find offset?
    // No, usually it's fixed layout based on capacity.
    // In melvin_minit, we did: header + nodes_cap * sizeof(Node) + edges_cap * sizeof(Edge)
    // So we need to use node_cap from header to calculate offset.
    g.edges = (Edge*)((uint8_t*)g.nodes + g.header->node_cap * sizeof(Node));

    printf("Melvin Runtime v2\n");
    printf("Nodes: %llu/%llu\n", g.header->num_nodes, g.header->node_cap);
    printf("Edges: %llu/%llu\n", g.header->num_edges, g.header->edge_cap);

    // Set stdin to non-blocking
    int flags = fcntl(0, F_GETFL, 0);
    fcntl(0, F_SETFL, flags | O_NONBLOCK);

    // Load all plugins
    MCFn mc_fs_seed = load_plugin_function("mc_fs", "mc_fs_seed");
    MCFn mc_fs_read_chunk = load_plugin_function("mc_fs", "mc_fs_read_chunk");
    MCFn mc_stdio_in = load_plugin_function("mc_io", "mc_stdio_in");
    MCFn mc_stdio_out = load_plugin_function("mc_io", "mc_stdio_out");
    MCFn mc_compile = load_plugin_function("mc_build", "mc_compile");
    MCFn mc_loader = load_plugin_function("mc_build", "mc_loader");
    MCFn mc_materialize_module_from_graph = load_plugin_function("mc_build", "mc_materialize_module_from_graph");
    MCFn mc_bootstrap_cog_module = load_plugin_function("mc_bootstrap", "mc_bootstrap_cog_module");
    MCFn mc_parse_c_file = load_plugin_function("mc_parse", "mc_parse_c_file");
    MCFn mc_process_scaffolds = load_plugin_function("mc_scaffold", "mc_process_scaffolds");
    
    // Register MC functions
    register_mc("zero", NULL);
    register_mc("fs_seed", mc_fs_seed);
    register_mc("fs_read", mc_fs_read_chunk);
    register_mc("stdio_in", mc_stdio_in);
    register_mc("stdio_out", mc_stdio_out);
    register_mc("compile", mc_compile);
    register_mc("loader", mc_loader);
    register_mc("materialize", mc_materialize_module_from_graph);
    register_mc("bootstrap_cog", mc_bootstrap_cog_module);
    register_mc("parse_c", mc_parse_c_file);
    register_mc("process_scaffolds", mc_process_scaffolds);
    
    // Create and activate scaffold processing node on startup
    if (mc_process_scaffolds) {
        uint64_t scaffold_node = alloc_node(&g);
        if (scaffold_node != UINT64_MAX && scaffold_node < g.header->node_cap) {
            g.nodes[scaffold_node].kind = NODE_KIND_CONTROL;
            // Find the MC ID for process_scaffolds
            for (uint32_t i = 0; i < g_mc_count; i++) {
                if (g_mc_table[i].name && strcmp(g_mc_table[i].name, "process_scaffolds") == 0) {
                    g.nodes[scaffold_node].mc_id = i;
                    g.nodes[scaffold_node].bias = 5.0f; // Activate on startup
                    g.nodes[scaffold_node].a = 1.0f;
                    printf("[main] Created scaffold processing node %llu\n", (unsigned long long)scaffold_node);
                    break;
                }
            }
        }
    }

    // Main loop
    while (1) {
        melvin_tick(&g);
        
        if (g.header->tick % 100 == 0) {
            printf("Tick %llu\r", g.header->tick);
            fflush(stdout);
        }
        
        usleep(1000); // 1ms sleep to prevent 100% CPU in loop
    }

    return 0;
}

