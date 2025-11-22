#include <stdio.h>
#include <stdlib.h>
#include "melvin.h"

void mc_beep(Brain *g, uint64_t node_id) {
    (void)g;
    (void)node_id;
    // Try to play a sound. Mac: afplay. Linux: aplay.
    // We are on Mac.
    system("afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1 &");
    fprintf(stderr, "[mc_beep] *PING* node=%llu\n", (unsigned long long)node_id);
}

