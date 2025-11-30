/* Load graph state from .m file (Mathematica format) - stub for now */
#include "melvin.h"
#include <stdio.h>
#include <string.h>

Graph* melvin_load_m(const char *filename) {
    /* For now, just init a new graph */
    /* Full parser would read nodes/edges from .m file */
    Graph *g = melvin_init();
    (void)filename; /* TODO: parse .m file */
    return g;
}

