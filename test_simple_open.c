#include "src/melvin.h"
#include <stdio.h>

int main() {
    fprintf(stderr, "Testing melvin_open...\n");
    Graph *g = melvin_open("test_simple.m", 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open\n");
        return 1;
    }
    fprintf(stderr, "Success! nodes=%llu edges=%llu\n", 
            (unsigned long long)g->node_count,
            (unsigned long long)g->edge_count);
    melvin_close(g);
    return 0;
}
