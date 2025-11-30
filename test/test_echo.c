/* Test 1: Simple echo - text in → text out */
#include "melvin.h"
#include <stdio.h>

int main(void) {
    Graph *g = melvin_init();
    uint32_t in = melvin_create_node(g, 0);
    uint32_t out = melvin_create_node(g, 0);
    
    /* Feed 'A' repeatedly */
    for (int i = 0; i < 10; i++) {
        melvin_feed_byte(g, in, 'A', 1.0f);
        for (int j = 0; j < 20; j++) melvin_step(g);
        
        uint8_t buf[256];
        size_t len = melvin_collect_output(g, out, buf, sizeof(buf));
        printf("Episode %d: output len=%zu\n", i, len);
    }
    
    melvin_free(g);
    return 0;
}
