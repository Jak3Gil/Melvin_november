#define _POSIX_C_SOURCE 200809L
#include "melvin.h"
#include <stdio.h>

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

