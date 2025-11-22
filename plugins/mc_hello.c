#include <stdio.h>
#include "melvin.h"

// This must match the MCFn signature: void fn(Brain *g, uint64_t node_id)
void mc_hello(Brain *g, uint64_t node_id) {
    (void)g;
    (void)node_id;
    fprintf(stderr, "[mc_hello] Hello from dynamic module! node=%llu\n",
            (unsigned long long)node_id);
}

