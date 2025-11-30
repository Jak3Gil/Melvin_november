/* Test 3: Residual context - A residual biases B */
#include "melvin.h"
#include <stdio.h>

int main(void) {
    Graph *g = melvin_init();
    uint32_t in = melvin_create_node(g, 0);
    
    /* Baseline: B with no context */
    for (int i = 0; i < 50; i++) melvin_step(g); /* Decay */
    melvin_feed_byte(g, in, 'B', 1.0f);
    for (int i = 0; i < 5; i++) melvin_step(g);
    float baseline = melvin_get_activation(g, 'B');
    
    /* Test: A then B quickly */
    for (int i = 0; i < 50; i++) melvin_step(g); /* Decay */
    melvin_feed_byte(g, in, 'A', 1.0f);
    for (int i = 0; i < 10; i++) melvin_step(g); /* Partial decay */
    float residual_a = melvin_get_activation(g, 'A');
    melvin_feed_byte(g, in, 'B', 1.0f);
    for (int i = 0; i < 5; i++) melvin_step(g);
    float context_b = melvin_get_activation(g, 'B');
    
    printf("Baseline B: %.4f\n", baseline);
    printf("Residual A: %.4f\n", residual_a);
    printf("Context B: %.4f (diff=%.4f)\n", context_b, context_b - baseline);
    
    melvin_free(g);
    return 0;
}
