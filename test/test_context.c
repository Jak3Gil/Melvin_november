/* Test 2: Context - A vs B form different flows */
#include "melvin.h"
#include <stdio.h>

int main(void) {
    Graph *g = melvin_init();
    uint32_t in = melvin_create_node(g, 0);
    uint32_t out = melvin_create_node(g, 0);
    
    /* Alternate A and B */
    for (int i = 0; i < 15; i++) {
        uint8_t b = (i % 2 == 0) ? 'A' : 'B';
        melvin_feed_byte(g, in, b, 1.0f);
        for (int j = 0; j < 15; j++) melvin_step(g);
        
        float a_a = melvin_get_activation(g, 'A');
        float a_b = melvin_get_activation(g, 'B');
        printf("Episode %d (%c): A=%.4f B=%.4f\n", i, b, a_a, a_b);
    }
    
    melvin_free(g);
    return 0;
}
