/*
 * melvin_run_continuous.c - Continuous runner for Melvin
 * 
 * Runs Melvin continuously, feeding data and triggering UEL propagation.
 * Can run for hours/days, learning from corpus and new inputs.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <math.h>
#include <errno.h>
#include <stdint.h>

static volatile int running = 1;

static void signal_handler(int sig) {
    (void)sig;
    running = 0;
    printf("\nShutting down...\n");
}

static void print_status(Graph *g, int iteration) {
    if (!g || !g->hdr) return;
    
    printf("[%d] Nodes: %llu | Edges: %llu | Chaos: %.6f | Activation: %.6f\n",
           iteration,
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count,
           g->avg_chaos,
           g->avg_activation);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <melvin.m file> [interval_seconds]\n", argv[0]);
        fprintf(stderr, "  interval: Seconds between UEL cycles (default: 1)\n");
        return 1;
    }
    
    const char *path = argv[1];
    int interval = (argc > 2) ? atoi(argv[2]) : 1;
    if (interval < 1) interval = 1;
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("Starting Melvin continuous runner\n");
    printf("Brain: %s\n", path);
    printf("Interval: %d seconds\n", interval);
    printf("Press Ctrl+C to stop\n\n");
    
    Graph *g = melvin_open(path, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", path);
        fprintf(stderr, "Error: %s\n", strerror(errno));
        perror("melvin_open");
        return 1;
    }
    
    /* Initialize syscalls */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    printf("Brain opened: %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    printf("Starting continuous run...\n\n");
    
    /* Bootstrap: Feed some initial input to get the system active */
    printf("Bootstrapping with initial input...\n");
    for (int i = 0; i < 10; i++) {
        uint8_t byte = (uint8_t)(i % 256);
        melvin_feed_byte(g, 0, byte, 0.2f);  /* Feed through port 0 with energy */
    }
    printf("Bootstrap complete. Starting continuous processing...\n\n");
    
    int iteration = 0;
    time_t last_sync = time(NULL);
    time_t last_feed = time(NULL);
    const int SYNC_INTERVAL = 60;  /* Sync every 60 seconds */
    const int FEED_INTERVAL = 5;   /* Feed new input every 5 seconds to keep it active */
    
    while (running) {
        /* Feed periodic input to keep the system active */
        time_t now = time(NULL);
        if (now - last_feed >= FEED_INTERVAL) {
            /* Feed a random byte to keep the graph active */
            uint8_t random_byte = (uint8_t)(rand() % 256);
            melvin_feed_byte(g, 0, random_byte, 0.2f);  /* Increased from 0.1f to 0.2f for stronger activation */
            last_feed = now;
        }
        
        /* Trigger UEL propagation - this runs wave propagation and exec nodes */
        melvin_call_entry(g);
        
        /* SELF-ACTIVATION: If output nodes are active, feed them back into working memory */
        /* This creates internal thinking loops - like human thought */
        if (g->node_count > 200) {
            for (uint32_t output = 100; output < 200 && output < g->node_count; output++) {
                float activation = fabsf(g->nodes[output].a);
                if (activation > 0.05f) {  /* Output node is active */
                    /* Feed back into working memory - creates self-activation */
                    uint32_t memory = 200 + (output % 10);
                    melvin_feed_byte(g, memory, (uint8_t)(output % 256), activation * 0.3f);  /* Feed with reduced energy */
                }
            }
        }
        
        /* Print status every 10 iterations */
        if (iteration % 10 == 0) {
            print_status(g, iteration);
        }
        
        /* Sync to disk periodically */
        if (now - last_sync >= SYNC_INTERVAL) {
            melvin_sync(g);
            last_sync = now;
            printf("[%d] Synced to disk\n", iteration);
        }
        
        iteration++;
        sleep(interval);
    }
    
    printf("\nFinal sync...\n");
    melvin_sync(g);
    
    printf("Stopped after %d iterations\n", iteration);
    print_status(g, iteration);
    
    melvin_close(g);
    return 0;
}

