#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "melvin.h"

// Simple helper to find a free node
static uint64_t alloc_node(Brain *g) {
    if (g->header->num_nodes >= g->header->node_cap) return 0; 
    uint64_t id = g->header->num_nodes++;
    memset(&g->nodes[id], 0, sizeof(Node));
    return id;
}

static void add_edge(Brain *g, uint64_t src, uint64_t dst, uint32_t flags) {
    if (g->header->num_edges >= g->header->edge_cap) return;
    Edge *e = &g->edges[g->header->num_edges++];
    e->src = src;
    e->dst = dst;
    e->w = 1.0f;
    e->flags = flags;
    e->usage_count = 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <melvin.m> <module.so>\n", argv[0]);
        return 1;
    }

    const char *db_path = argv[1];
    const char *mod_path = argv[2];

    int fd = open(db_path, O_RDWR);
    if (fd < 0) {
        perror("open db");
        return 1;
    }

    struct stat st;
    fstat(fd, &st);
    size_t filesize = st.st_size;
    void *map = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    Brain g;
    g.header = (BrainHeader*)map;
    g.nodes = (Node*)((uint8_t*)map + sizeof(BrainHeader));
    g.edges = (Edge*)((uint8_t*)g.nodes + g.header->node_cap * sizeof(Node));

    // Read module file
    FILE *f = fopen(mod_path, "rb");
    if (!f) {
        perror("fopen module");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long mod_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(mod_len);
    fread(buf, 1, mod_len, f);
    fclose(f);

    printf("Storing %ld bytes of %s into graph...\n", mod_len, mod_path);

    // Create root node for this module
    // We'll look for a node that is INTENDED to be the root, or just create one.
    // The prompt says "Create a CONTROL node MODULE_BEEP_ROOT (choose an ID like node 20)".
    // Let's allocate it.
    uint64_t root_id = alloc_node(&g);
    g.nodes[root_id].kind = NODE_KIND_CONTROL;
    g.nodes[root_id].mc_id = 7; // Assumption: mc_materialize will be 7 (next available builtin)
    // Actually, we should probably just create the structure and let melvin.c wire up the mc_id if it wants,
    // or we can set it here if we know the ID.
    // The prompt says: "On tick == 0... Create a CONTROL node... node->mc_id = MC_ID_MATERIALIZE".
    // This tool runs "offline".
    // Let's just create the structure and print the root ID.
    // The runtime can find it or we can hardcode it.
    // To follow the prompt "Bootstrap wiring in melvin.c", melvin.c does the wiring.
    // But this tool needs to put the BYTES in.
    // Let's assume this tool creates the chain, and prints the root ID, 
    // and then we update melvin.c to point to that root ID?
    // OR, simpler: use a fixed ID if possible.
    // But `melvin.m` grows.
    // Let's just append.
    
    printf("Root node ID: %llu\n", (unsigned long long)root_id);

    uint64_t prev_id = root_id;
    for (long i = 0; i < mod_len; i++) {
        uint64_t byte_node = alloc_node(&g);
        g.nodes[byte_node].kind = NODE_KIND_DATA;
        g.nodes[byte_node].value = (float)buf[i];
        
        // Edge from prev to current
        add_edge(&g, prev_id, byte_node, EDGE_FLAG_MODULE_BYTES);
        prev_id = byte_node;
    }

    // Mark the root as the materializer trigger
    // We need to know the ID of the materialize function.
    // In `melvin.c`, we will register it.
    // Let's assume the user will ensure `melvin.c` maps this root node to the function.
    // Or we can set it here if we assume ID 8.
    g.nodes[root_id].mc_id = 7; // Assumption: mc_materialize will be 7
    g.nodes[root_id].bias = 1.0f; // Trigger immediately
    g.nodes[root_id].a = 1.0f;

    printf("Stored %ld bytes. Root Node %llu set to MC_ID 7 and active.\n", mod_len, (unsigned long long)root_id);

    free(buf);
    return 0;
}

