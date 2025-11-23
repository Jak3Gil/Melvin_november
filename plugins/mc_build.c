#define _POSIX_C_SOURCE 200809L
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>

// External MC table and helpers (defined in melvin.c)
typedef void (*MCFn)(Brain *, uint64_t);

typedef struct {
    const char *name;
    MCFn fn;
    uint32_t flags;
} MCEntry;

extern MCEntry g_mc_table[];
extern uint32_t g_mc_count;
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

#define MAX_MC_FUNCS 256

// Track compiled files to avoid recompilation
static char **compiled_files = NULL;
static size_t compiled_count = 0;
static size_t compiled_cap = 0;

static int is_file_compiled(const char *path) {
    for (size_t i = 0; i < compiled_count; i++) {
        if (strcmp(compiled_files[i], path) == 0) {
            return 1;
        }
    }
    return 0;
}

static void mark_file_compiled(const char *path) {
    if (is_file_compiled(path)) return;
    
    if (compiled_count >= compiled_cap) {
        compiled_cap = compiled_cap ? compiled_cap * 2 : 64;
        compiled_files = realloc(compiled_files, compiled_cap * sizeof(char*));
    }
    compiled_files[compiled_count++] = strdup(path);
}

// Find all .c files that need compilation
static void find_c_files_to_compile(const char *dir_path, char ***found_files, size_t *found_count, size_t *found_cap) {
    DIR *dir = opendir(dir_path);
    if (!dir) return;
    
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        
        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name);
        
        struct stat st;
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
            size_t len = strlen(path);
            if (len > 2 && strcmp(path + len - 2, ".c") == 0) {
                // Check if .so exists and is newer
                char so_path[2048];
                snprintf(so_path, sizeof(so_path), "%.*s.so", (int)(len - 2), path);
                
                struct stat so_st;
                int needs_compile = 1;
                if (stat(so_path, &so_st) == 0) {
                    // .so exists, check if .c is newer
                    if (st.st_mtime <= so_st.st_mtime) {
                        needs_compile = 0; // .so is up to date
                    }
                }
                
                if (needs_compile && !is_file_compiled(path)) {
                    if (*found_count >= *found_cap) {
                        *found_cap = *found_cap ? *found_cap * 2 : 128;
                        *found_files = realloc(*found_files, *found_cap * sizeof(char*));
                    }
                    (*found_files)[(*found_count)++] = strdup(path);
                }
            }
        } else if (S_ISDIR(st.st_mode)) {
            // Skip .git, node_modules, build
            if (strcmp(ent->d_name, ".git") != 0 && 
                strcmp(ent->d_name, "node_modules") != 0 &&
                strcmp(ent->d_name, "build") != 0) {
                find_c_files_to_compile(path, found_files, found_count, found_cap);
            }
        }
    }
    closedir(dir);
}

void mc_compile(Brain *g, uint64_t node_id) {
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }

    // Find all .c files that need compilation
    char **found_files = NULL;
    size_t found_count = 0;
    size_t found_cap = 0;
    
    // Check plugins/ directory (where plugin .c files live)
    find_c_files_to_compile("./plugins", &found_files, &found_count, &found_cap);
    
    if (found_count == 0) {
        fprintf(stderr, "[mc_compile] No .c files need compilation.\n");
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        return;
    }
    
    fprintf(stderr, "[mc_compile] Found %zu .c files to compile...\n", found_count);
    
    int compiled_any = 0;
    
    // Compile each .c file
    for (size_t i = 0; i < found_count; i++) {
        const char *c_file = found_files[i];
        
        // Generate .so path
        char so_path[2048];
        size_t len = strlen(c_file);
        snprintf(so_path, sizeof(so_path), "%.*s.so", (int)(len - 2), c_file);
        
        fprintf(stderr, "[mc_compile] Compiling %s -> %s...\n", c_file, so_path);
        
        // Build compile command
        char cmd[4096];
        snprintf(cmd, sizeof(cmd), 
                 "clang -shared -fPIC -O2 -I. -undefined dynamic_lookup -o %s %s",
                 so_path, c_file);
        
        int rc = system(cmd);
        
        if (rc == 0) {
            fprintf(stderr, "[mc_compile] ✓ %s compiled successfully\n", c_file);
            
            // Create compilation node in graph to track this compilation
            uint64_t compile_node = alloc_node(g);
            if (compile_node != UINT64_MAX) {
                Node *cn = &g->nodes[compile_node];
                cn->kind = NODE_KIND_META;
                cn->a = 0.8f;
                cn->value = 0x434F4D50; // "COMP"
                
                // Link to compile node
                add_edge(g, node_id, compile_node, 1.0f, EDGE_FLAG_CONTROL);
            }
            
            mark_file_compiled(c_file);
            compiled_any = 1;
        } else {
            fprintf(stderr, "[mc_compile] ✗ Compilation failed for %s (exit code %d)\n", c_file, rc);
        }
    }
    
    // Free found_files list
    for (size_t i = 0; i < found_count; i++) {
        free(found_files[i]);
    }
    free(found_files);
    
    if (compiled_any) {
        fprintf(stderr, "[mc_compile] Compilation complete. New modules ready to load.\n");
    }
    
    // Deactivate compile node
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

void mc_loader(Brain *g, uint64_t node_id) {
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

