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

static MCEntry  g_mc_table[MAX_MC_FUNCS];
static uint32_t g_mc_count = 0;

// ========================================================
// Helper Functions
// ========================================================

// Find a free node
static uint64_t alloc_node(Brain *g) {
    if (g->header->num_nodes >= g->header->node_cap) return 0; // 0 is usually reserved or valid, but let's check
    // Assume append for now
    uint64_t id = g->header->num_nodes++;
    memset(&g->nodes[id], 0, sizeof(Node));
    return id;
}

// Add edge
static void add_edge(Brain *g, uint64_t src, uint64_t dst, float w, uint32_t flags) {
    if (g->header->num_edges >= g->header->edge_cap) return;
    
    // Simple linear scan to update existing edge? 
    // For speed, let's just append. Compaction is for later.
    // Or check if exists? Expensive for O(E).
    // We'll just append for the seed/bootstrap phase.
    
    Edge *e = &g->edges[g->header->num_edges++];
    e->src = src;
    e->dst = dst;
    e->w = w;
    e->flags = flags;
    e->usage_count = 1;
}

// ========================================================
// MC Functions
// ========================================================

// Static state for FS exploration
static char **file_list = NULL;
static size_t file_count = 0;
static size_t file_cap = 0;
static size_t current_file_idx = 0;
static FILE *current_fp = NULL;
static uint64_t current_file_node_id = 0;

static void add_file_to_list(const char *path) {
    if (file_count >= file_cap) {
        file_cap = file_cap ? file_cap * 2 : 1024;
        file_list = realloc(file_list, file_cap * sizeof(char*));
    }
    file_list[file_count++] = strdup(path);
}

static void scan_dir_recursive(const char *base_path) {
    DIR *dir = opendir(base_path);
    if (!dir) return;

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", base_path, ent->d_name);
        
        struct stat st;
        if (stat(path, &st) == 0) {
            if (S_ISREG(st.st_mode)) {
                add_file_to_list(path);
            } else if (S_ISDIR(st.st_mode)) {
                scan_dir_recursive(path);
            }
        }
    }
    closedir(dir);
}

void mc_fs_seed(Brain *g, uint64_t node_id) {
    static int seeded = 0;
    if (seeded) return;

    printf("[MC] Seeding filesystem...\n");
    scan_dir_recursive("."); // Scan current dir for now, or "./corpus" if exists
    
    // Also try ./data and ./corpus specifically if they exist
    scan_dir_recursive("./data");
    scan_dir_recursive("./corpus");

    printf("[MC] Found %zu files.\n", file_count);

    // Create nodes for files (optional, or just track internally)
    // For now, we just have the list.
    
    seeded = 1;
}

void mc_fs_read_chunk(Brain *g, uint64_t node_id) {
    if (file_count == 0) return;
    if (current_file_idx >= file_count) current_file_idx = 0; // Loop or stop? Loop for now.

    if (!current_fp) {
        const char *path = file_list[current_file_idx];
        current_fp = fopen(path, "rb");
        if (!current_fp) {
            // Failed to open, skip
            current_file_idx++;
            return;
        }
        printf("[MC] Reading file: %s\n", path);
        // Create a node representing this file if we want
    }

    uint8_t buffer[4096];
    size_t n = fread(buffer, 1, sizeof(buffer), current_fp);
    
    if (n > 0) {
        static uint64_t last_byte_node = 0;
        
        for (size_t i = 0; i < n; i++) {
            uint8_t b = buffer[i];
            // Activate node 0-255
            if (b < g->header->num_nodes) { // Assuming first 256 nodes are bytes
                 g->nodes[b].a = 1.0f;
                 g->nodes[b].value = (float)b;
                 g->nodes[b].kind = NODE_KIND_DATA;
                 
                 if (last_byte_node != 0) {
                     // Add sequence edge
                     // This is simplistic; in real Melvin we check for existence
                     add_edge(g, last_byte_node, b, 1.0f, EDGE_FLAG_SEQ);
                 }
                 last_byte_node = b;
            }
        }
    }

    if (n < sizeof(buffer)) {
        // EOF or Error
        fclose(current_fp);
        current_fp = NULL;
        current_file_idx++;
    }
}

void mc_stdio_in(Brain *g, uint64_t node_id) {
    int c = getchar();
    if (c != EOF) {
        // Activate node representing this byte?
        if (c >= 0 && c < 256) {
             g->nodes[c].a = 1.0f;
             // Add edge from this I/O node to... where?
             // For now just input injection.
        }
    }
}

void mc_stdio_out(Brain *g, uint64_t node_id) {
    Node *n = &g->nodes[node_id];
    // If this node is active, output its value as char?
    // Or if it's an OUTPUT node.
    // The prompt says: "Let them read/write single bytes from stdin/stdout."
    // Usually this MC function is attached to a specific node.
    // If this node is active, print something.
    // Maybe print the value of the strongest connected node? 
    // Or just print a fixed character if it's a "Print 'A'" node?
    // Let's assume it prints the char corresponding to the node's value if valid.
    if (n->value >= 0 && n->value < 256) {
        putchar((int)n->value);
        fflush(stdout);
    }
}

void mc_compile(Brain *g, uint64_t node_id) {
    static int compiled = 0;
    if (compiled) {
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        return;
    }

    fprintf(stderr, "[mc_compile] Compiling plugins/mc_beep.c...\n");
    int rc = system("clang -shared -fPIC -O2 -I. -o plugins/mc_beep.so plugins/mc_beep.c");
    
    if (rc == 0) {
        fprintf(stderr, "[mc_compile] Compilation success!\n");
        // Trigger loader?
        // For this test, we assume loader is running independently or we activate it here.
        // Let's activate Node 261 (Loader).
        if (g->header->num_nodes > 261) {
            g->nodes[261].bias = 5.0f;
            g->nodes[261].a = 1.0f;
        }
    } else {
        fprintf(stderr, "[mc_compile] Compilation failed: %d\n", rc);
    }
    
    compiled = 1;
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

void mc_loader(Brain *g, uint64_t node_id) {
    // ... (existing implementation)
    static int loaded = 0;
    if (loaded) {
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        return;
    }

    // Try loading mc_beep first
    const char *lib = "plugins/mc_beep.so";
    const char *sym = "mc_beep";
    
    if (access(lib, F_OK) != 0) {
        return; // Not ready yet
    }

    fprintf(stderr, "[mc_loader] Loading %s\n", lib);

    void *h = dlopen(lib, RTLD_NOW);
    if (!h) {
        fprintf(stderr, "[mc_loader] dlopen failed: %s\n", dlerror());
        return;
    }

    MCFn fn = (MCFn)dlsym(h, sym);
    if (!fn) {
        fprintf(stderr, "[mc_loader] dlsym failed: %s\n", dlerror());
        return;
    }

    // Register
    uint32_t id = g_mc_count++;
    g_mc_table[id].name  = strdup(sym);
    g_mc_table[id].fn    = fn;
    g_mc_table[id].flags = 0;
    
    fprintf(stderr, "[mc_loader] registered %s as mc_id=%u\n", sym, id);

    // Assign to Node 11 and activate (but with low bias to prevent continuous beeping)
    Node *target = &g->nodes[11];
    target->mc_id = id;
    target->bias = -5.0f; // Disabled by default - graph can activate it later
    target->a = 0.0f;
    
    loaded = 1;
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

void mc_materialize_module_from_graph(Brain *g, uint64_t node_id) {
    static int materialized = 0;
    if (materialized) {
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        return;
    }

    fprintf(stderr, "[mc_materialize] Extracting module from graph (root=%llu)...\n", (unsigned long long)node_id);

    const char *temp_path = "/tmp/mc_beep_from_graph.so";
    FILE *f = fopen(temp_path, "wb");
    if (!f) {
        perror("fopen temp");
        return;
    }

    // Traverse edges
    uint64_t curr = node_id;
    size_t total_bytes = 0;
    
    // Find first child
    uint64_t next = UINT64_MAX;
    uint64_t edge_cap = g->header->edge_cap; // Use cap to be safe or count?
    // Actually we should use num_edges
    uint64_t num_edges = g->header->num_edges;

    for(uint64_t i=0; i<num_edges; i++) {
        Edge *e = &g->edges[i];
        if (e->src == curr && (e->flags & EDGE_FLAG_MODULE_BYTES)) {
            next = e->dst;
            break;
        }
    }
    
    curr = next;
    
    while (curr != UINT64_MAX && curr < g->header->num_nodes) {
        // Write byte
        uint8_t b = (uint8_t)g->nodes[curr].value;
        fputc(b, f);
        total_bytes++;
        
        // Find next
        uint64_t prev = curr;
        next = UINT64_MAX;
        for(uint64_t i=0; i<num_edges; i++) {
            Edge *e = &g->edges[i];
            if (e->src == prev && (e->flags & EDGE_FLAG_MODULE_BYTES)) {
                next = e->dst;
                break;
            }
        }
        curr = next;
        
        if (total_bytes > 1000000) { // Safety break
            fprintf(stderr, "[mc_materialize] Safety break: module too large\n");
            break;
        }
    }
    
    fclose(f);
    fprintf(stderr, "[mc_materialize] wrote %zu bytes to %s\n", total_bytes, temp_path);
    
    // Ad-hoc sign the binary for macOS
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "codesign -s - %s", temp_path);
    system(cmd);

    // Now load it
    void *h = dlopen(temp_path, RTLD_NOW);
    if (!h) {
        fprintf(stderr, "[mc_materialize] dlopen failed: %s\n", dlerror());
        return;
    }
    
    MCFn fn = (MCFn)dlsym(h, "mc_beep");
    if (!fn) {
        fprintf(stderr, "[mc_materialize] dlsym failed: %s\n", dlerror());
        return;
    }
    
    uint32_t id = g_mc_count++;
    g_mc_table[id].name = "mc_beep";
    g_mc_table[id].fn = fn;
    g_mc_table[id].flags = 0;
    
    fprintf(stderr, "[mc_materialize] registered mc_beep as mc_id=%u\n", id);
    
    // Assign to Node 12 and activate it (but with low bias to prevent continuous beeping)
    uint64_t target_id = 12; 
    if (target_id < g->header->num_nodes) {
        g->nodes[target_id].mc_id = id;
        g->nodes[target_id].bias = -5.0f; // Disabled by default - graph can activate it later
        g->nodes[target_id].a = 0.0f;
        fprintf(stderr, "[mc_materialize] Registered Node %llu with mc_id=%u (inactive)\n", (unsigned long long)target_id, id);
    }

    materialized = 1;
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

// ========================================================
// Core Runtime
// ========================================================

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
        // Debug print for node 0
        // if (i == 0) {
        //      fprintf(stderr, "Node 0: mc_id=%u a=%f\n", n->mc_id, n->a);
        // }
        if (n->a < 0.5f) continue; // Lowered threshold for testing

        // Adjust for 0-based index vs 1-based ID? 
        // Prompt says "0 = none; >0 = index". So index is mc_id.
        // But we need to check bounds.
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
    // In a real system, this would pull from buffers filled by MC nodes or external threads
    // For now, MC nodes (mc_fs_read) write directly to node activations.
    // So this might be a no-op if MC nodes did the work, or we might clamp inputs.
}

void propagate_predictions(Brain *g) {
    ensure_buffers(g);
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;

    // 1. Reset predictions
    for(uint64_t i=0; i<n; i++) {
        g_predicted_a[i] = g->nodes[i].bias; // Start with bias
    }

    // 2. Sum weighted inputs
    // Note: This is O(E). 
    for(uint64_t i=0; i<e_count; i++) {
        Edge *e = &g->edges[i];
        if (e->src < n && e->dst < n) {
            float input = e->w * g->nodes[e->src].a;
            g_predicted_a[e->dst] += input;
        }
    }

    // 3. Apply activation function
    for(uint64_t i=0; i<n; i++) {
        g_predicted_a[i] = sigmoid(g_predicted_a[i]);
    }
    
    // 4. Update internal nodes (Reality Update)
    // Input nodes (kind=DATA) might be clamped by ingest_input/MC nodes, 
    // so we only update nodes that weren't externally driven?
    // For simplicity, we blend: a = (1-alpha)*a + alpha*pred
    float alpha = 0.2f;
    for(uint64_t i=0; i<n; i++) {
        Node *node = &g->nodes[i];
        // If node was strongly driven by MC/Input (a=1.0), keep it?
        // Or blend. Let's blend.
        node->a = (1.0f - alpha) * node->a + alpha * g_predicted_a[i];
    }
}

void apply_environment(Brain *g) {
    // Physics/Env constraints. 
    // Currently empty.
}

void compute_error(Brain *g) {
    ensure_buffers(g);
    uint64_t n = g->header->num_nodes;
    
    for(uint64_t i=0; i<n; i++) {
        // Error = Actual - Predicted
        // We just updated Actual (node->a) in propagate/ingest.
        // Predicted is g_predicted_a.
        g_node_error[i] = g->nodes[i].a - g_predicted_a[i];
    }
}

void update_edges(Brain *g) {
    uint64_t e_count = g->header->num_edges;
    float lambda = 0.9f; // Trace decay
    float eta = 0.01f;   // Learning rate

    for(uint64_t i=0; i<e_count; i++) {
        Edge *e = &g->edges[i];
        if (e->src >= g->header->num_nodes || e->dst >= g->header->num_nodes) continue;
        
        Node *src = &g->nodes[e->src];
        Node *dst = &g->nodes[e->dst];

        // 1. Update Eligibility
        // elig(t+1) = lambda * elig(t) + src.a * dst.a
        e->elig = lambda * e->elig + (src->a * dst->a);

        // 2. Calculate Influence (Simplified)
        // influence ~ weight * src.a
        float influence = e->w * src->a;
        
        // 3. Update Weight
        // dw = eta * error[dst] * influence * elig
        // Using g_node_error[dst]
        float err = g_node_error[e->dst];
        float dw = eta * err * influence * e->elig;
        
        e->w += dw;
        
        // Clamp weights
        if (e->w > 10.0f) e->w = 10.0f;
        if (e->w < -10.0f) e->w = -10.0f;
        
        // Update usage
        if (fabsf(src->a * dst->a) > 0.1f) {
            e->usage_count++;
        }
    }
}

void induce_patterns(Brain *g) {
    // Stub: Real pattern induction would analyze g_node_error and g_predicted_a history
    // to create new nodes for repeating sequences.
}

void emit_output(Brain *g) {
    // Stub
}

void melvin_tick(Brain *g) {
    ingest_input(g);          // stub ok
    propagate_predictions(g); // stub ok
    apply_environment(g);     // stub ok
    compute_error(g);         // stub ok
    update_edges(g);          // stub ok
    induce_patterns(g);       // stub or empty
    
    run_mc_nodes(g);
    
    emit_output(g);           // stub ok
    g->header->tick++;
    
    // Decay activations
    for(uint64_t i=0; i<g->header->num_nodes; i++) {
        g->nodes[i].a *= 0.9f; // Simple decay
    }
}

int main(int argc, char **argv) {
    const char *db_path = "melvin.m";
    if (argc > 1) db_path = argv[1];

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

    // Register MC functions
    // Using 1-based indices for mc_id usually implies 0 is null.
    // But array is 0-based.
    // Let's just say mc_id 1 = index 1.
    register_mc("zero", NULL); // Slot 0 (unused)
    register_mc("fs_seed", mc_fs_seed);
    register_mc("fs_read", mc_fs_read_chunk);
    register_mc("stdio_in", mc_stdio_in);
    register_mc("stdio_out", mc_stdio_out);
    register_mc("compile", mc_compile);
    register_mc("loader", mc_loader); // mc_id = 6
    register_mc("materialize", mc_materialize_module_from_graph); // mc_id = 7

    // Ensure we have a node for fs_seed and fs_read
    // If brain is empty (num_nodes == 0), let's create some bootstrap nodes.
    if (g.header->num_nodes < 262) {
        printf("Bootstrapping nodes...\n");
        // Ensure nodes 0-255 exist
        while(g.header->num_nodes < 256) {
            uint64_t id = alloc_node(&g);
            g.nodes[id].kind = NODE_KIND_DATA;
            g.nodes[id].value = (float)id;
        }
        
        // Node 256: FS Seed
        uint64_t seed_node = alloc_node(&g);
        g.nodes[seed_node].kind = NODE_KIND_CONTROL;
        g.nodes[seed_node].mc_id = 1; // fs_seed
        g.nodes[seed_node].a = 1.0f; // Activate to start
        printf("Created FS Seed node %llu\n", seed_node);

        // Node 257: FS Read
        uint64_t read_node = alloc_node(&g);
        g.nodes[read_node].kind = NODE_KIND_CONTROL;
        g.nodes[read_node].mc_id = 2; // fs_read
        g.nodes[read_node].a = 1.0f; // Activate to start
        printf("Created FS Read node %llu\n", read_node);
        
        // Node 258, 259, 260 reserved...
        while(g.header->num_nodes < 261) alloc_node(&g);

        // Node 261: Loader (initially inactive)
        uint64_t loader_node = alloc_node(&g);
        g.nodes[loader_node].kind = NODE_KIND_CONTROL;
        g.nodes[loader_node].mc_id = 6; // loader
        g.nodes[loader_node].bias = -5.0f; 
        g.nodes[loader_node].a = 0.0f;
        printf("Created Loader node %llu\n", loader_node);

        // Node 262: Compiler (initially inactive)
        uint64_t compiler_node = alloc_node(&g);
        g.nodes[compiler_node].kind = NODE_KIND_CONTROL;
        g.nodes[compiler_node].mc_id = 5; // compile
        g.nodes[compiler_node].bias = -5.0f; 
        g.nodes[compiler_node].a = 0.0f;
        printf("Created Compiler node %llu\n", compiler_node);
    } else {
        // Restart logic: No automatic re-trigger
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
