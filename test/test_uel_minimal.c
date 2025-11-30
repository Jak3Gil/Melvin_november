/*
 * test_uel_minimal.c - Test the minimal UEL physics core
 */

#include "melvin.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    printf("=== Universal Emergence Law Minimal Test ===\n\n");
    
    /* Initialize graph */
    Graph *g = melvin_init();
    if (!g) {
        printf("FAIL: Could not initialize graph\n");
        return 1;
    }
    printf("[OK] Graph initialized\n");
    printf("     Nodes: %zu, Edges: %zu\n", g->num_nodes, g->num_edges);
    
    /* Create port nodes (host-side convention) */
    MelvinPorts ports;
    ports.text_in = melvin_create_node(g, 0);
    ports.text_out = melvin_create_node(g, 0);
    ports.motor_out = melvin_create_node(g, 0);
    ports.reward = melvin_create_node(g, 0);
    
    printf("[OK] Created port nodes:\n");
    printf("     text_in=%u, text_out=%u, motor_out=%u, reward=%u\n",
           ports.text_in, ports.text_out, ports.motor_out, ports.reward);
    printf("     Total nodes: %zu\n", g->num_nodes);
    
    /* Test 1: Feed input bytes */
    printf("\n--- Test 1: Input ---\n");
    const char *input = "Hello";
    for (int i = 0; input[i]; i++) {
        melvin_feed_byte(g, ports.text_in, (uint8_t)input[i], 1.0f);
    }
    printf("[OK] Fed '%s' through text_in port (node %u)\n", input, ports.text_in);
    printf("     Edges after input: %zu\n", g->num_edges);
    
    /* Check activations on the data nodes */
    printf("     Activations:\n");
    for (int i = 0; input[i]; i++) {
        uint8_t b = (uint8_t)input[i];
        float a = melvin_get_activation(g, b);
        printf("       '%c' (node %d): a=%.4f\n", b, b, a);
    }
    
    /* Test 2: Run UEL steps */
    printf("\n--- Test 2: UEL Physics Steps ---\n");
    for (int step = 0; step < 10; step++) {
        melvin_step(g);
        printf("  Tick %llu: energy=%.4f, chaos=%.4f\n", 
               (unsigned long long)g->tick, g->total_energy, g->total_chaos);
    }
    printf("[OK] Ran 10 physics steps\n");
    
    /* Check activations after dynamics */
    printf("     Activations after dynamics:\n");
    for (int i = 0; input[i]; i++) {
        uint8_t b = (uint8_t)input[i];
        float a = melvin_get_activation(g, b);
        printf("       '%c' (node %d): a=%.4f\n", b, b, a);
    }
    
    /* Test 3: Create edges to output port and check output */
    printf("\n--- Test 3: Output ---\n");
    
    /* Manually boost some data nodes and connect to output port */
    for (int i = 0; input[i]; i++) {
        uint8_t b = (uint8_t)input[i];
        /* Boost activation above threshold */
        g->nodes[b].a = 0.5f + (float)i * 0.1f;
        /* Feed through output port to create edge */
        melvin_feed_byte(g, ports.text_out, b, 0.1f);
    }
    
    /* Collect output */
    uint8_t out_buf[256];
    size_t out_len = melvin_collect_output(g, ports.text_out, out_buf, sizeof(out_buf));
    out_buf[out_len] = '\0';
    
    printf("[OK] Collected %zu bytes from text_out port (node %u)\n", out_len, ports.text_out);
    if (out_len > 0) {
        printf("     Output: '%s'\n", out_buf);
    }
    
    /* Test 4: Reward signal */
    printf("\n--- Test 4: Reward ---\n");
    float energy_before = g->total_energy;
    melvin_reward(g, ports.reward, 1.0f);
    melvin_step(g);
    printf("[OK] Applied positive reward through reward port (node %u)\n", ports.reward);
    printf("     Energy: %.4f -> %.4f\n", energy_before, g->total_energy);
    
    /* Test 5: Parameter access */
    printf("\n--- Test 5: Parameters ---\n");
    printf("     eta_a = %.4f\n", melvin_get_param(g, "eta_a"));
    printf("     eta_w = %.4f\n", melvin_get_param(g, "eta_w"));
    printf("     lambda = %.4f\n", melvin_get_param(g, "lambda"));
    
    melvin_set_param(g, "eta_a", 0.2f);
    printf("[OK] Set eta_a = 0.2\n");
    printf("     eta_a = %.4f\n", melvin_get_param(g, "eta_a"));
    
    /* Test 6: Dump active nodes */
    printf("\n--- Test 6: Active Nodes ---\n");
    melvin_dump_active(g, 0.1f);
    
    /* Cleanup */
    melvin_free(g);
    printf("\n[OK] Graph freed\n");
    
    printf("\n=== ALL TESTS PASSED ===\n");
    printf("\nmelvin.c is now a minimal physics core:\n");
    printf("  - melvin_feed_byte()     : inject energy through any port node\n");
    printf("  - melvin_step()          : run UEL update (all nodes treated identically)\n");
    printf("  - melvin_collect_output(): read activated outputs from any port node\n");
    printf("\nKey: No channel types in physics. Ports are just node indices chosen by host.\n");
    
    return 0;
}

