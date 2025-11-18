#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>  // for STDOUT_FILENO
// Note: legacy_learning functions are now in melvin.c

// Timing helper: get current time in milliseconds (monotonic clock)
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

// Simple JSON output helper
static void print_json_string(const char *s) {
    putchar('"');
    for (const char *p = s; *p; p++) {
        if (*p == '"' || *p == '\\') putchar('\\');
        putchar(*p);
    }
    putchar('"');
}

static void print_json_float(float f) {
    printf("%.4f", f);
}

// Endurance mode configuration
typedef struct {
    const char *input_path;
    const char *snapshot_path;
    uint64_t snapshot_interval_tasks;
    uint64_t snapshot_interval_seconds;
    bool training_enabled;
    uint64_t max_runtime_seconds;  // Maximum runtime (0 = unlimited)
    uint64_t metrics_interval_tasks;  // Log metrics every N tasks
} EnduranceConfig;

// Count patterns in graph (for metrics)
static uint64_t count_patterns(const Graph *g) {
    if (!g) return 0;
    uint64_t count = 0;
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].kind == NODE_PATTERN) {
            count++;
        }
    }
    return count;
}

// Log endurance metrics
static void log_endurance_metrics(Graph *g, uint64_t task_count, double start_time_ms) {
    double now = now_ms();
    double elapsed_s = (now - start_time_ms) / 1000.0;
    double tasks_per_s = (elapsed_s > 0.0) ? (double)task_count / elapsed_s : 0.0;
    
    uint64_t num_patterns = count_patterns(g);
    
    fprintf(stderr,
        "[endurance] tasks=%llu elapsed=%.1fs tps=%.2f "
        "nodes=%llu edges=%llu patterns=%llu\n",
        (unsigned long long)task_count,
        elapsed_s,
        tasks_per_s,
        (unsigned long long)g->num_nodes,
        (unsigned long long)g->num_edges,
        (unsigned long long)num_patterns);
}

// Find latest snapshot file in directory (or return path if it's a file)
static char *find_latest_snapshot(const char *snapshot_path) {
    if (!snapshot_path) return NULL;
    
    struct stat st;
    if (stat(snapshot_path, &st) != 0) {
        return NULL;  // Path doesn't exist
    }
    
    // If it's a file, return it
    if (S_ISREG(st.st_mode)) {
        char *result = malloc(strlen(snapshot_path) + 1);
        if (result) strcpy(result, snapshot_path);
        return result;
    }
    
    // If it's a directory, find newest .snap file
    if (S_ISDIR(st.st_mode)) {
        DIR *dir = opendir(snapshot_path);
        if (!dir) return NULL;
        
        char *latest = NULL;
        time_t latest_time = 0;
        
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".snap") == NULL) continue;
            
            char full_path[1024];
            snprintf(full_path, sizeof(full_path), "%s/%s", snapshot_path, entry->d_name);
            
            if (stat(full_path, &st) == 0 && S_ISREG(st.st_mode)) {
                if (st.st_mtime > latest_time) {
                    latest_time = st.st_mtime;
                    free(latest);
                    latest = malloc(strlen(full_path) + 1);
                    if (latest) strcpy(latest, full_path);
                }
            }
        }
        
        closedir(dir);
        return latest;
    }
    
    return NULL;
}

// Rotate snapshots: keep last K snapshots, delete older ones
static void rotate_snapshots(const char *snapshot_path, int keep_count) {
    if (!snapshot_path || keep_count <= 0) return;
    
    // Extract directory and base name
    char dir[1024] = {0};
    char base[256] = {0};
    const char *last_slash = strrchr(snapshot_path, '/');
    if (last_slash) {
        size_t dir_len = last_slash - snapshot_path;
        strncpy(dir, snapshot_path, dir_len);
        strcpy(base, last_slash + 1);
    } else {
        strcpy(base, snapshot_path);
    }
    
    // For simplicity, just keep the main snapshot file
    // Full rotation can be added later if needed
    (void)keep_count;  // Unused for now
}

// Process a single task (input chunk) - simplified version for endurance mode
static void process_task(Graph *g, const uint8_t *data, size_t len) {
    if (!g || !data || len == 0) return;
    
    // Feed data as DATA nodes
    uint64_t prev_data_id = UINT32_MAX;
    for (size_t i = 0; i < len; i++) {
        Node *data_node = graph_add_data_byte(g, data[i]);
        if (data_node) {
            if (prev_data_id != UINT32_MAX) {
                graph_add_edge(g, prev_data_id, data_node->id, 1.0f);
            }
            prev_data_id = data_node->id;
        }
    }
    
    // Create patterns from bigrams and trigrams (limited for performance)
    Node *patterns[16];
    size_t num_patterns = 0;
    
    // Generate patterns for bigrams
    for (size_t i = 0; i < len - 1 && num_patterns < 8; i++) {
        PatternAtom atoms[2];
        atoms[0].delta = 0;
        atoms[0].mode = 0;
        atoms[0].value = data[i];
        atoms[1].delta = 1;
        atoms[1].mode = 0;
        atoms[1].value = data[i+1];
        
        Node *pattern = graph_add_pattern(g, atoms, 2, 0.5f);
        if (pattern) {
            patterns[num_patterns++] = pattern;
        }
    }
    
    // Generate patterns for trigrams
    for (size_t i = 0; i < len - 2 && num_patterns < 16; i++) {
        PatternAtom atoms[3];
        atoms[0].delta = 0;
        atoms[0].mode = 0;
        atoms[0].value = data[i];
        atoms[1].delta = 1;
        atoms[1].mode = 0;
        atoms[1].value = data[i+1];
        atoms[2].delta = 2;
        atoms[2].mode = 0;
        atoms[2].value = data[i+2];
        
        Node *pattern = graph_add_pattern(g, atoms, 3, 0.5f);
        if (pattern) {
            patterns[num_patterns++] = pattern;
        }
    }
    
    // Only run learning if training is enabled and we have patterns
    extern SystemConfig g_sys;
    if (g_sys.training_enabled && num_patterns > 0 && len > 2) {
        uint64_t input_start = g->next_data_pos > len ? g->next_data_pos - len : 0;
        uint64_t input_end = g->next_data_pos > 0 ? g->next_data_pos - 1 : 0;
        
        // Collect connected patterns (graph-native, dynamic)
        Node *connected_patterns[64];
        size_t num_connected = 0;
        
        for (uint64_t data_id = input_start; data_id <= input_end && num_connected < 64; data_id++) {
            Node *data_node = graph_find_node_by_id(g, data_id);
            if (!data_node || data_node->kind != NODE_DATA) continue;
            
            uint32_t eid = data_node->first_in_edge;
            uint32_t visited = 0;
            while (eid != UINT32_MAX && eid < g->num_edges && visited < 100) {
                visited++;
                Edge *e = &g->edges[eid];
                Node *src = graph_find_node_by_id(g, e->src);
                
                if (src && src->kind == NODE_PATTERN) {
                    int found = 0;
                    for (size_t i = 0; i < num_connected; i++) {
                        if (connected_patterns[i] == src) {
                            found = 1;
                            break;
                        }
                    }
                    if (!found) {
                        connected_patterns[num_connected++] = src;
                    }
                }
                eid = e->next_in_edge;
            }
        }
        
        // Add newly created patterns
        for (size_t p = 0; p < num_patterns && num_connected < 64; p++) {
            int found = 0;
            for (size_t i = 0; i < num_connected; i++) {
                if (connected_patterns[i] == patterns[p]) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                connected_patterns[num_connected++] = patterns[p];
            }
        }
        
        // Time-bounded learning: only 1 iteration for endurance mode
        if (num_connected > 0 && input_end >= input_start) {
            graph_self_consistency_episode_multi_pattern(g,
                                                         connected_patterns,
                                                         num_connected,
                                                         input_start,
                                                         input_end,
                                                         0.8f,
                                                         0.2f);
        }
    }
}

// Run endurance mode
static int run_endurance_mode(const EnduranceConfig *cfg) {
    if (!cfg) return 1;
    
    fprintf(stderr, "[endurance] Starting endurance mode\n");
    fprintf(stderr, "[endurance] Input: %s\n", cfg->input_path);
    fprintf(stderr, "[endurance] Snapshot: %s\n", cfg->snapshot_path);
    fprintf(stderr, "[endurance] Training: %s\n", cfg->training_enabled ? "enabled" : "disabled");
    fprintf(stderr, "[endurance] Snapshot interval: %llu tasks or %llu seconds\n",
            (unsigned long long)cfg->snapshot_interval_tasks,
            (unsigned long long)cfg->snapshot_interval_seconds);
    
    // Load or create graph
    double t0 = now_ms();
    Graph *g = NULL;
    char *snapshot_to_load = find_latest_snapshot(cfg->snapshot_path);
    
    if (snapshot_to_load) {
        g = graph_load_auto(snapshot_to_load);
        if (g) {
            fprintf(stderr, "[endurance] Loaded graph from: %s\n", snapshot_to_load);
        }
        free(snapshot_to_load);
    }
    
    if (!g) {
        fprintf(stderr, "[endurance] No snapshot found, starting with new graph\n");
        g = graph_create(1024, 2048, 16 * 1024);
    }
    
    double t1 = now_ms();
    fprintf(stderr, "[endurance] Graph load took %.1f ms\n", t1 - t0);
    
    if (!g) {
        fprintf(stderr, "[endurance] ERROR: failed to create graph\n");
        return 1;
    }
    
    // Set training mode
    extern SystemConfig g_sys;
    g_sys.training_enabled = cfg->training_enabled ? 1 : 0;
    
    // Open input file
    FILE *input_file = fopen(cfg->input_path, "rb");
    if (!input_file) {
        fprintf(stderr, "[endurance] ERROR: failed to open input file: %s\n", cfg->input_path);
        graph_destroy(g);
        return 1;
    }
    
    // Initialize counters
    uint64_t task_count = 0;
    uint64_t last_snapshot_task = 0;
    double start_time = now_ms();
    double last_snapshot_time = start_time;
    
    // Endurance loop
    uint8_t buffer[4096];  // 4KB chunks
    bool should_stop = false;
    
    fprintf(stderr, "[endurance] Starting processing loop...\n");
    
    while (!should_stop) {
        // Read chunk from input
        size_t bytes_read = fread(buffer, 1, sizeof(buffer), input_file);
        
        if (bytes_read == 0) {
            if (feof(input_file)) {
                fprintf(stderr, "[endurance] Reached end of input file\n");
                break;
            }
            if (ferror(input_file)) {
                fprintf(stderr, "[endurance] Error reading input file\n");
                break;
            }
            continue;
        }
        
        // Process this chunk as a task
        process_task(g, buffer, bytes_read);
        task_count++;
        
        // Check if we should stop (max runtime)
        double now = now_ms();
        if (cfg->max_runtime_seconds > 0) {
            double elapsed_s = (now - start_time) / 1000.0;
            if (elapsed_s >= cfg->max_runtime_seconds) {
                fprintf(stderr, "[endurance] Max runtime reached (%.1f seconds)\n", elapsed_s);
                should_stop = true;
            }
        }
        
        // Log metrics periodically
        if (task_count % cfg->metrics_interval_tasks == 0) {
            log_endurance_metrics(g, task_count, start_time);
        }
        
        // Check if we need to snapshot
        bool need_snapshot = false;
        if (task_count - last_snapshot_task >= cfg->snapshot_interval_tasks) {
            need_snapshot = true;
        }
        if ((now - last_snapshot_time) / 1000.0 >= cfg->snapshot_interval_seconds) {
            need_snapshot = true;
        }
        
        if (need_snapshot) {
            double snap_t0 = now_ms();
            if (graph_save_snapshot(g, cfg->snapshot_path)) {
                double snap_t1 = now_ms();
                fprintf(stderr,
                    "[endurance] snapshot saved to %s in %.2f ms (tasks=%llu)\n",
                    cfg->snapshot_path, snap_t1 - snap_t0,
                    (unsigned long long)task_count);
                
                // Rotate snapshots
                rotate_snapshots(cfg->snapshot_path, 5);
            } else {
                fprintf(stderr, "[endurance] snapshot save FAILED: %s\n", cfg->snapshot_path);
            }
            
            last_snapshot_task = task_count;
            last_snapshot_time = now;
        }
    }
    
    // Final snapshot
    fprintf(stderr, "[endurance] Saving final snapshot...\n");
    graph_save_snapshot(g, cfg->snapshot_path);
    
    // Final metrics
    double total_time = (now_ms() - start_time) / 1000.0;
    log_endurance_metrics(g, task_count, start_time);
    fprintf(stderr, "[endurance] Completed: %llu tasks in %.1f seconds\n",
            (unsigned long long)task_count, total_time);
    
    fclose(input_file);
    graph_destroy(g);
    
    return 0;
}

// Process a single input string through the graph
static void process_input(Graph *g, const char *buffer, size_t len) {
    if (len == 0) {
        printf("{\"error\": \"empty input\"}\n");
        return;
    }
    
    // Feed string as DATA nodes
    uint64_t prev_data_id = UINT32_MAX;
    for (size_t i = 0; i < len; i++) {
        Node *data_node = graph_add_data_byte(g, (uint8_t)buffer[i]);
        if (data_node) {
            if (prev_data_id != UINT32_MAX) {
                graph_add_edge(g, prev_data_id, data_node->id, 1.0f);
            }
            prev_data_id = data_node->id;
        }
    }
    
    // Create patterns from common bigrams and trigrams
    Node *patterns[16];
    size_t num_patterns = 0;
    
    // Generate patterns for all bigrams and trigrams in the input
    for (size_t i = 0; i < len - 1 && num_patterns < 16; i++) {
        // Bigram
        char bigram[3] = {buffer[i], buffer[i+1], '\0'};
        
        PatternAtom atoms[2];
        atoms[0].delta = 0;
        atoms[0].mode = 0;
        atoms[0].value = (uint8_t)bigram[0];
        atoms[1].delta = 1;
        atoms[1].mode = 0;
        atoms[1].value = (uint8_t)bigram[1];
        
        Node *pattern = graph_add_pattern(g, atoms, 2, 0.5f);
        if (pattern) {
            patterns[num_patterns++] = pattern;
        }
    }
    
    // Also try trigrams
    for (size_t i = 0; i < len - 2 && num_patterns < 16; i++) {
        char trigram[4] = {buffer[i], buffer[i+1], buffer[i+2], '\0'};
        
        PatternAtom atoms[3];
        atoms[0].delta = 0;
        atoms[0].mode = 0;
        atoms[0].value = (uint8_t)trigram[0];
        atoms[1].delta = 1;
        atoms[1].mode = 0;
        atoms[1].value = (uint8_t)trigram[1];
        atoms[2].delta = 2;
        atoms[2].mode = 0;
        atoms[2].value = (uint8_t)trigram[2];
        
        Node *pattern = graph_add_pattern(g, atoms, 3, 0.5f);
        if (pattern) {
            patterns[num_patterns++] = pattern;
        }
    }
    
    if (num_patterns == 0) {
        printf("{\"error\": \"no patterns created\"}\n");
        return;
    }
    
    // Run learning episodes - DYNAMIC: only use patterns connected to input
    // Use the actual input length, not entire history
    // This is intelligent: we only learn on what we just fed, not old data
    uint64_t input_start = g->next_data_pos > len ? g->next_data_pos - len : 0;
    uint64_t input_end = g->next_data_pos > 0 ? g->next_data_pos - 1 : 0;
    
    // Collect patterns that are actually connected to input data nodes (graph-native)
    // This is dynamic computation - only process what's connected, not scan everything
    Node *connected_patterns[64];
    size_t num_connected = 0;
    
    // Start from input data nodes and traverse connections to find patterns
    for (uint64_t data_id = input_start; data_id <= input_end && num_connected < 64; data_id++) {
        Node *data_node = graph_find_node_by_id(g, data_id);
        if (!data_node || data_node->kind != NODE_DATA) continue;
        
        // Traverse incoming edges to find patterns
        uint32_t eid = data_node->first_in_edge;
        uint32_t visited = 0;
        while (eid != UINT32_MAX && eid < g->num_edges && visited < 100) {
            visited++;
            Edge *e = &g->edges[eid];
            Node *src = graph_find_node_by_id(g, e->src);
            
            if (src && src->kind == NODE_PATTERN) {
                // Check if already in list
                int found = 0;
                for (size_t i = 0; i < num_connected; i++) {
                    if (connected_patterns[i] == src) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    connected_patterns[num_connected++] = src;
                }
            }
            eid = e->next_in_edge;
        }
    }
    
    // Also include patterns we just created (they might not have edges yet)
    for (size_t p = 0; p < num_patterns && num_connected < 64; p++) {
        int found = 0;
        for (size_t i = 0; i < num_connected; i++) {
            if (connected_patterns[i] == patterns[p]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            connected_patterns[num_connected++] = patterns[p];
        }
    }
    
    // Only run learning if we have connected patterns and data
    // Truly dynamic: skip learning for very short inputs (they're already simple)
    // Only learn when there's actual complexity to process
    if (input_end >= input_start && len > 0 && num_connected > 0 && len > 2) {
        // Very minimal learning - just 1 iteration for most cases
        // The graph learns incrementally over time, not all at once
        int learning_iterations = (len <= 4) ? 1 : (len <= 10) ? 2 : 3;
        
        for (int it = 0; it < learning_iterations; it++) {
            graph_self_consistency_episode_multi_pattern(g,
                                                         connected_patterns,
                                                         num_connected,
                                                         input_start,
                                                         input_end,
                                                         0.8f,  // match_threshold
                                                         0.2f); // lr_q
        }
    }
    
    // Use same range for explanation building
    uint64_t start_id = input_start;
    uint64_t end_id = input_end;
    
    // Build final explanation - only use connected patterns
    Explanation candidates, selected;
    explanation_init(&candidates);
    explanation_init(&selected);
    
    graph_collect_candidates_multi_pattern(g, connected_patterns, num_connected,
                                          start_id, end_id,
                                          0.8f, &candidates);
    
    explanation_select_greedy_consistent(g, &candidates,
                                         start_id, end_id, &selected);
    
    // Reconstruct and compute metrics
    size_t segment_len = (size_t)(end_id - start_id + 1);
    uint8_t *actual = malloc(segment_len);
    uint8_t *pred = malloc(segment_len);
    
    float compression_ratio = 0.0f;
    float reconstruction_error = 0.0f;
    
    if (actual && pred) {
        graph_collect_data_span(g, start_id, actual, segment_len);
        graph_reconstruct_from_explanation(g, &selected,
                                          start_id, end_id,
                                          pred, segment_len);
        
        size_t positions = 0;
        size_t errors = 0;
        for (size_t i = 0; i < segment_len; i++) {
            if (pred[i] == 0x00) continue;
            positions++;
            if (pred[i] != actual[i]) {
                errors++;
            }
        }
        
        if (segment_len > 0) {
            compression_ratio = (float)selected.count / (float)segment_len;
        }
        
        if (positions > 0) {
            reconstruction_error = (float)errors / (float)positions;
        }
    }
    
    // Graph-native output generation (using OUTPUT nodes)
    // Step 0: Create OUTPUT nodes from learned patterns (if they don't exist)
    // This is graph-native: we're just creating nodes and edges, not changing C hardware
    // Use byte value as direct index (true O(1) lookup, no hard limit)
    // Since bytes are 0-255, we can index directly: output_cache[byte] = node
    Node *output_cache[256] = {0};  // Direct index by byte value (NULL = not found)
    int cache_built = 0;
    
    for (size_t p = 0; p < num_patterns; p++) {
        Node *pattern = patterns[p];
        
        // For each pattern, create an OUTPUT node that represents the pattern's "response"
        // The output byte is derived from the first atom of the pattern
        if (pattern->payload_len >= sizeof(PatternAtom)) {
            const PatternAtom *atoms = (const PatternAtom *)(g->blob + pattern->payload_offset);
            if (atoms[0].mode == 0) {  // CONST_BYTE
                uint8_t output_byte = atoms[0].value;
                
                // Build cache once (lazy initialization - only scan OUTPUT nodes if needed)
                if (!cache_built) {
                    // Scan OUTPUT nodes once and index by byte value (O(N) but only once)
                    for (uint64_t i = 0; i < g->num_nodes; i++) {
                        Node *n = &g->nodes[i];
                        if (n->kind == NODE_OUTPUT && n->payload_len > 0) {
                            uint8_t b = g->blob[n->payload_offset];
                            output_cache[b] = n;  // Direct index by byte value
                        }
                    }
                    cache_built = 1;
                }
                
                // True O(1) lookup using byte as direct index
                Node *output_node = output_cache[output_byte];
                
                // Create OUTPUT node if it doesn't exist
                if (!output_node) {
                    // Graph-native: create OUTPUT node using hardware function
                    // In future, patterns will decide to create OUTPUT nodes
                    static uint64_t next_output_id = (1ULL << 62);
                    uint8_t payload = output_byte;
                    output_node = graph_create_node(g, NODE_OUTPUT, next_output_id++, &payload, 1);
                    if (output_node) {
                        output_cache[output_byte] = output_node;  // Cache it
                    }
                }
                
                // Create edge from pattern to OUTPUT node (if it doesn't exist)
                if (output_node) {
                    // Check if edge already exists (use adjacency list - O(out_degree))
                    uint32_t eid = pattern->first_out_edge;
                    int edge_exists = 0;
                    while (eid != UINT32_MAX && eid < g->num_edges) {
                        Edge *e = &g->edges[eid];
                        if (e->dst == output_node->id) {
                            edge_exists = 1;
                            break;
                        }
                        eid = e->next_out_edge;
                    }
                    
                    // Create edge if it doesn't exist
                    if (!edge_exists) {
                        graph_add_edge(g, pattern->id, output_node->id, pattern->q);  // Weight = pattern quality
                    }
                }
            }
        }
    }
    
    // Step 1: Activate patterns dynamically - use graph connections, not scanning
    // Start from input data nodes and traverse their connections to find patterns
    // This is truly dynamic: O(connections_to_input) not O(patterns × input_length)
    
    // Track which patterns we've seen and their best scores
    typedef struct {
        Node *pattern;
        float max_score;
        uint64_t best_anchor;
    } PatternActivation;
    PatternActivation pattern_activations[16] = {0};
    size_t num_activations = 0;
    
    // For each input data node, traverse its incoming edges to find patterns
    // Use the input_start/input_end already defined above
    // This is graph-native: we follow connections, not scan everything
    for (uint64_t data_id = input_start; data_id <= input_end; data_id++) {
        Node *data_node = graph_find_node_by_id(g, data_id);
        if (!data_node || data_node->kind != NODE_DATA) continue;
        
        // Traverse incoming edges to find patterns connected to this data node
        uint32_t eid = data_node->first_in_edge;
        uint32_t visited = 0;
        while (eid != UINT32_MAX && eid < g->num_edges && visited < 100) {
            visited++;
            Edge *e = &g->edges[eid];
            Node *src = graph_find_node_by_id(g, e->src);
            
            // If source is a pattern, check if it matches at this anchor
            if (src && src->kind == NODE_PATTERN) {
                // Check if we already have this pattern
                size_t p_idx = SIZE_MAX;
                for (size_t i = 0; i < num_activations; i++) {
                    if (pattern_activations[i].pattern == src) {
                        p_idx = i;
                        break;
                    }
                }
                
                // If new pattern, add it
                if (p_idx == SIZE_MAX && num_activations < 16) {
                    p_idx = num_activations++;
                    pattern_activations[p_idx].pattern = src;
                    pattern_activations[p_idx].max_score = 0.0f;
                    pattern_activations[p_idx].best_anchor = UINT64_MAX;
                }
                
                // Check match score at this anchor
                if (p_idx != SIZE_MAX) {
                    float score = pattern_match_score(g, src, data_id);
                    if (score > pattern_activations[p_idx].max_score) {
                        pattern_activations[p_idx].max_score = score;
                        pattern_activations[p_idx].best_anchor = data_id;
                    }
                }
            }
            
            eid = e->next_in_edge;
        }
    }
    
    // Activate patterns we found
    for (size_t i = 0; i < num_activations; i++) {
        pattern_activations[i].pattern->a = pattern_activations[i].max_score * 
                                            pattern_activations[i].pattern->q;
    }
    
    // Also activate patterns we just created (they might not have edges yet)
    for (size_t p = 0; p < num_patterns; p++) {
        Node *pattern = patterns[p];
        // Check if already activated
        int found = 0;
        for (size_t i = 0; i < num_activations; i++) {
            if (pattern_activations[i].pattern == pattern) {
                found = 1;
                break;
            }
        }
        // If not found via connections, check match at input start
        if (!found) {
            float score = pattern_match_score(g, pattern, input_start);
            pattern->a = score * pattern->q;
        }
    }
    
    // Step 2: Graph-native error feedback - teach the graph that "no output" is wrong
    // This creates internal consequences for failures - fully graph-native
    // OUTPUT nodes are NOT special - they're just nodes that happen to be outputs
    
    // Propagate activation from patterns to OUTPUT nodes
    graph_propagate(g, 3);
    
    // Run graph-native learning rules (triggered by active LEARNER nodes)
    graph_run_local_rules(g);
    
    // Emit output from active OUTPUT nodes (graph-native output generation)
    graph_emit_output(g, 4, STDOUT_FILENO);  // Use default max_output_bytes_per_tick = 4
    
    // Count active OUTPUT nodes using cache (O(256) = O(1) since bytes are 0-255)
    size_t active_output_count = 0;
    for (int i = 0; i < 256; i++) {
        Node *n = output_cache[i];
        if (n && n->a > 0.1f) {
            active_output_count++;
        }
    }
    
    // Graph-native learning: if no outputs, this is a failure state
    // Strengthen edges from active patterns to OUTPUT nodes
    // This teaches: "when patterns activate, outputs should activate"
    if (active_output_count == 0) {
        // Failure state - update edge weights to strengthen pattern->output connections
        for (size_t p = 0; p < num_patterns; p++) {
            Node *pattern = patterns[p];
            if (pattern->a > 0.1f) {  // Pattern was active but didn't produce output
                // Strengthen edges from pattern to OUTPUT nodes (use adjacency list - O(out_degree))
                uint32_t eid = pattern->first_out_edge;
                while (eid != UINT32_MAX && eid < g->num_edges) {
                    Edge *e = &g->edges[eid];
                    Node *dst = graph_find_node_by_id(g, e->dst);
                    if (dst && dst->kind == NODE_OUTPUT) {
                        // Negative error = "should have fired but didn't" = strengthen
                        float error = -1.0f * pattern->a;
                        graph_update_edge_from_error(g, pattern->id, dst->id, error, 0.5f);
                    }
                    eid = e->next_out_edge;
                }
            }
        }
        
        // Re-propagate after weight updates
        graph_propagate(g, 2);
        
        // Re-check outputs using cache
        active_output_count = 0;
        for (int i = 0; i < 256; i++) {
            Node *n = output_cache[i];
            if (n && n->a > 0.1f) {
                active_output_count++;
            }
        }
    }
    
    // Step 4: Collect OUTPUT nodes with activation > threshold using cache
    char graph_output[256] = {0};
    size_t output_len = 0;
    float output_threshold = 0.1f;  // Threshold for output generation
    
    // Sort OUTPUT nodes by activation (descending)
    typedef struct {
        Node *node;
        float activation;
    } OutputCandidate;
    
    OutputCandidate output_candidates[256];
    size_t num_output_candidates = 0;
    
    // Use cache instead of scanning all nodes (O(256) = O(1) since bytes are 0-255)
    for (int i = 0; i < 256 && num_output_candidates < 256; i++) {
        Node *n = output_cache[i];
        if (n && n->a > output_threshold && n->payload_len > 0) {
            output_candidates[num_output_candidates].node = n;
            output_candidates[num_output_candidates].activation = n->a;
            num_output_candidates++;
        }
    }
    
    // Simple bubble sort by activation (descending)
    for (size_t i = 0; i < num_output_candidates; i++) {
        for (size_t j = i + 1; j < num_output_candidates; j++) {
            if (output_candidates[j].activation > output_candidates[i].activation) {
                OutputCandidate tmp = output_candidates[i];
                output_candidates[i] = output_candidates[j];
                output_candidates[j] = tmp;
            }
        }
    }
    
    // Collect output bytes
    for (size_t i = 0; i < num_output_candidates && output_len < sizeof(graph_output) - 1; i++) {
        Node *out_node = output_candidates[i].node;
        if (out_node->payload_len > 0 && out_node->payload_offset < g->blob_used) {
            uint8_t b = g->blob[out_node->payload_offset];
            graph_output[output_len++] = (char)b;
        }
    }
    graph_output[output_len] = '\0';
    
    // Output JSON
    printf("{\n");
    printf("  \"input_str\": ");
    print_json_string(buffer);
    printf(",\n");
    printf("  \"segment_len\": %zu,\n", segment_len);
    printf("  \"num_patterns\": %zu,\n", num_patterns);
    printf("  \"explanation_apps\": %zu,\n", selected.count);
    printf("  \"compression_ratio\": ");
    print_json_float(compression_ratio);
    printf(",\n");
    printf("  \"reconstruction_error\": ");
    print_json_float(reconstruction_error);
    printf(",\n");
    printf("  \"graph_output\": ");
    if (output_len > 0) {
        print_json_string(graph_output);
    } else {
        printf("\"(no output - no OUTPUT nodes active)\"");
    }
    printf(",\n");
    printf("  \"patterns\": [\n");
    
    for (size_t i = 0; i < num_patterns; i++) {
        Node *p = patterns[i];
        printf("    {\n");
        printf("      \"id\": %llu,\n", (unsigned long long)p->id);
        printf("      \"q\": ");
        print_json_float(p->q);
        printf(",\n");
        
        // Count bindings for this pattern using adjacency list (local, not global scan)
        size_t binding_count = 0;
        uint32_t eid = p->first_out_edge;
        uint32_t visited_count = 0;
        uint32_t max_visit = 1024;  // Debug safety limit
        
        while (eid != UINT32_MAX) {
            if (eid >= g->num_edges) break;  // Safety check
            if (visited_count++ > max_visit) {
                #ifdef DEBUG
                fprintf(stderr, "ERROR: Possible cycle in pattern %llu adjacency list\n", 
                        (unsigned long long)p->id);
                #endif
                break;
            }
            
            Edge *edge = &g->edges[eid];
            Node *d = graph_find_node_by_id(g, edge->dst);
            if (d && d->kind == NODE_DATA) {
                binding_count++;
            }
            eid = edge->next_out_edge;
        }
        
        printf("      \"binding_count\": %zu\n", binding_count);
        printf("    }%s\n", (i < num_patterns - 1) ? "," : "");
    }
    
    printf("  ]\n");
    printf("}\n");
    
    free(actual);
    free(pred);
    explanation_free(&candidates);
    explanation_free(&selected);
}

// Persistent main loop: process stdin line-by-line without reloading graph
static void run_main_loop(Graph *g, const char *save_file) {
    char buffer[1024];
    
    while (fgets(buffer, sizeof(buffer), stdin)) {
        // Remove newline
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
            len--;
        }
        
        if (len == 0) {
            continue;  // Skip empty lines
        }
        
        // Process this input
        process_input(g, buffer, len);
        
        // Flush output
        fflush(stdout);
    }
    
    // Save graph on exit if requested (use fast snapshot format)
    if (save_file) {
        graph_save_snapshot(g, save_file);
    }
}

int main(int argc, char *argv[]) {
    // Check for endurance mode first
    bool endurance_mode = false;
    EnduranceConfig endurance_cfg = {0};
    
    // Default endurance settings
    endurance_cfg.snapshot_interval_tasks = 1000;
    endurance_cfg.snapshot_interval_seconds = 300;  // 5 minutes
    endurance_cfg.training_enabled = false;  // Default: runtime mode
    endurance_cfg.max_runtime_seconds = 0;  // 0 = unlimited
    endurance_cfg.metrics_interval_tasks = 100;
    
    const char *load_file = NULL;
    const char *save_file = NULL;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--endurance") == 0 || strcmp(argv[i], "--8h-test") == 0) {
            endurance_mode = true;
        } else if (strcmp(argv[i], "--input-file") == 0 && i + 1 < argc) {
            endurance_cfg.input_path = argv[++i];
        } else if (strcmp(argv[i], "--snapshot-path") == 0 && i + 1 < argc) {
            endurance_cfg.snapshot_path = argv[++i];
        } else if (strcmp(argv[i], "--snapshot-interval-tasks") == 0 && i + 1 < argc) {
            endurance_cfg.snapshot_interval_tasks = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--snapshot-interval-seconds") == 0 && i + 1 < argc) {
            endurance_cfg.snapshot_interval_seconds = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--training-enabled") == 0) {
            endurance_cfg.training_enabled = true;
        } else if (strcmp(argv[i], "--runtime-only") == 0) {
            endurance_cfg.training_enabled = false;
        } else if (strcmp(argv[i], "--max-runtime-seconds") == 0 && i + 1 < argc) {
            endurance_cfg.max_runtime_seconds = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--metrics-interval-tasks") == 0 && i + 1 < argc) {
            endurance_cfg.metrics_interval_tasks = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            load_file = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_file = argv[++i];
        }
    }
    
    // If endurance mode, run it and exit
    if (endurance_mode) {
        if (!endurance_cfg.input_path) {
            fprintf(stderr, "[endurance] ERROR: --input-file required\n");
            return 1;
        }
        if (!endurance_cfg.snapshot_path) {
            fprintf(stderr, "[endurance] ERROR: --snapshot-path required\n");
            return 1;
        }
        return run_endurance_mode(&endurance_cfg);
    }
    
    // Load or create graph ONCE at startup (with timing)
    double t0 = now_ms();
    Graph *g = NULL;
    int is_loaded_graph = 0;
    if (load_file) {
        // Use auto-load: tries fast snapshot format first, falls back to legacy
        g = graph_load_auto(load_file);
        if (g) {
            is_loaded_graph = 1;  // Pre-trained graph - use fast inference mode
        } else {
            // If load fails, create new graph
            g = graph_create(1024, 2048, 16 * 1024);
        }
    } else {
        g = graph_create(1024, 2048, 16 * 1024);
    }
    double t1 = now_ms();
    
    if (!g) {
        fprintf(stderr, "[melvin] ERROR: failed to create graph\n");
        return 1;
    }
    
    // Log graph load time to stderr
    fprintf(stderr, "[melvin] graph_load_auto took %.1f ms\n", t1 - t0);
    
    // Training mode: enable only for new graphs or when explicitly training
    // Pre-trained graphs default to fast inference (no heavy learning)
    extern SystemConfig g_sys;
    if (is_loaded_graph) {
        g_sys.training_enabled = 0;  // Runtime mode: fast inference
    } else {
        g_sys.training_enabled = 1;  // Training mode: allow learning
    }
    
    // Run persistent main loop (with timing)
    double t_loop_start = now_ms();
    run_main_loop(g, save_file);
    double t_loop_end = now_ms();
    
    // Log main loop time to stderr
    fprintf(stderr, "[melvin] main loop took %.1f ms\n", t_loop_end - t_loop_start);
    
    graph_destroy(g);
    
    return 0;
}
