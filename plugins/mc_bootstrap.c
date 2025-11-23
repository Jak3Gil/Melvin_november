#define _POSIX_C_SOURCE 200809L
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>

// External MC table and helper (defined in melvin.c)
typedef void (*MCFn)(Brain *, uint64_t);

typedef struct {
    const char *name;
    MCFn fn;
    uint32_t flags;
} MCEntry;

extern MCEntry g_mc_table[];
extern uint32_t g_mc_count;

#define MAX_MC_FUNCS 256

// Helper to allocate node (inline since we can't link against melvin.c)
static uint64_t alloc_node_inline(Brain *g) {
    if (g->header->num_nodes >= g->header->node_cap) return UINT64_MAX;
    uint64_t id = g->header->num_nodes++;
    memset(&g->nodes[id], 0, sizeof(Node));
    return id;
}

void mc_bootstrap_cog_module(Brain *g, uint64_t node_id) {
    
    static int bootstrapped = 0;
    if (bootstrapped) {
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        return;
    }
    
    fprintf(stderr, "[mc_bootstrap_cog] Compiling and loading cognitive module...\n");
    
    // Check if already compiled
    if (access("plugins/mc_cog.so", F_OK) != 0) {
        // Compile mc_cog.c
        fprintf(stderr, "[mc_bootstrap_cog] Compiling plugins/mc_cog.c...\n");
        int rc = system("clang -shared -fPIC -O2 -I.. -o plugins/mc_cog.so plugins/mc_cog.c -lm");
        if (rc != 0) {
            fprintf(stderr, "[mc_bootstrap_cog] Compilation failed: %d\n", rc);
            return;
        }
        fprintf(stderr, "[mc_bootstrap_cog] Compilation success!\n");
    }
    
    // Load the module
    const char *lib = "plugins/mc_cog.so";
    void *h = dlopen(lib, RTLD_NOW);
    if (!h) {
        fprintf(stderr, "[mc_bootstrap_cog] dlopen failed: %s\n", dlerror());
        return;
    }
    
    // Get function pointers
    typedef void (*MCFn)(Brain *, uint64_t);
    MCFn mc_cog_tick = (MCFn)dlsym(h, "mc_cog_tick");
    MCFn mc_audio_in = (MCFn)dlsym(h, "mc_audio_in");
    MCFn mc_vision_in = (MCFn)dlsym(h, "mc_vision_in");
    
    if (!mc_cog_tick || !mc_audio_in || !mc_vision_in) {
        fprintf(stderr, "[mc_bootstrap_cog] dlsym failed: %s\n", dlerror());
        dlclose(h);
        return;
    }
    
    // Register in MC table
    if (g_mc_count + 3 >= MAX_MC_FUNCS) {
        fprintf(stderr, "[mc_bootstrap_cog] MC table full!\n");
        dlclose(h);
        return;
    }
    
    uint32_t id_cog = g_mc_count++;
    g_mc_table[id_cog].name = "mc_cog_tick";
    g_mc_table[id_cog].fn = (MCFn)mc_cog_tick;
    g_mc_table[id_cog].flags = 0;
    
    uint32_t id_audio = g_mc_count++;
    g_mc_table[id_audio].name = "mc_audio_in";
    g_mc_table[id_audio].fn = (MCFn)mc_audio_in;
    g_mc_table[id_audio].flags = 0;
    
    uint32_t id_vision = g_mc_count++;
    g_mc_table[id_vision].name = "mc_vision_in";
    g_mc_table[id_vision].fn = (MCFn)mc_vision_in;
    g_mc_table[id_vision].flags = 0;
    
    fprintf(stderr, "[mc_bootstrap_cog] Registered mc_cog_tick=%u, mc_audio_in=%u, mc_vision_in=%u\n",
            id_cog, id_audio, id_vision);
    
    // Create CONTROL nodes for these functions
    uint64_t cog_node = alloc_node_inline(g);
    g->nodes[cog_node].kind = NODE_KIND_CONTROL;
    g->nodes[cog_node].mc_id = id_cog;
    g->nodes[cog_node].bias = 0.5f; // Tend to fire regularly
    g->nodes[cog_node].a = 0.0f;
    
    uint64_t audio_node = alloc_node_inline(g);
    g->nodes[audio_node].kind = NODE_KIND_CONTROL;
    g->nodes[audio_node].mc_id = id_audio;
    g->nodes[audio_node].bias = -5.0f; // Low bias, graph can activate
    g->nodes[audio_node].a = 0.0f;
    
    uint64_t vision_node = alloc_node_inline(g);
    g->nodes[vision_node].kind = NODE_KIND_CONTROL;
    g->nodes[vision_node].mc_id = id_vision;
    g->nodes[vision_node].bias = -5.0f; // Low bias, graph can activate
    g->nodes[vision_node].a = 0.0f;
    
    fprintf(stderr, "[mc_bootstrap_cog] Created CONTROL nodes: cog=%llu, audio=%llu, vision=%llu\n",
            (unsigned long long)cog_node, (unsigned long long)audio_node, (unsigned long long)vision_node);
    
    // Mark as bootstrapped (store in a META node)
    uint64_t bootstrap_flag = alloc_node_inline(g);
    g->nodes[bootstrap_flag].kind = NODE_KIND_META;
    g->nodes[bootstrap_flag].value = 0x424F4F54; // "BOOT"
    g->nodes[bootstrap_flag].a = 1.0f;
    
    bootstrapped = 1;
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}
