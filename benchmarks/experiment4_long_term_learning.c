/*
 * Experiment 4: Long-Term Learning Efficiency
 * 
 * Question: How does Melvin's efficiency evolve over HOURS/DAYS of learning?
 * 
 * Hypothesis: Efficiency improves continuously as pattern library grows
 * 
 * Test Design:
 * 1. Feed diverse data continuously (books, articles, varied patterns)
 * 2. Measure efficiency every 1000 inputs
 * 3. Track: pattern growth rate, reuse ratio, memory efficiency
 * 4. Run for: 10K, 100K, 1M inputs (hours to days)
 * 
 * Expected: Learning rate INCREASES over time (opposite of ML!)
 * 
 * This simulates real-world usage: continuous learning over weeks/months
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/melvin.h"

/* Diverse text corpus for realistic learning */
const char *CORPUS[] = {
    /* Simple patterns - learned first */
    "the", "and", "for", "are", "but", "not", "you", "all",
    
    /* Common words - build vocabulary */
    "hello", "world", "system", "learning", "pattern", "neural",
    "computer", "science", "artificial", "intelligence",
    
    /* Sentences - compositional */
    "the quick brown fox jumps over the lazy dog",
    "all your base are belong to us",
    "to be or not to be that is the question",
    "i think therefore i am",
    
    /* Technical - domain specific */
    "neural network backpropagation gradient descent",
    "machine learning deep learning reinforcement learning",
    "pattern recognition hierarchical composition",
    "event driven graph neural architecture",
    
    /* Complex compositions */
    "the neural network uses backpropagation for gradient descent",
    "machine learning systems require large datasets for training",
    "hierarchical pattern composition enables exponential efficiency",
    "event driven architectures scale better than batch processing",
    
    NULL  /* Sentinel */
};

typedef struct {
    uint64_t input_count;
    uint64_t total_nodes;
    uint64_t total_edges;
    int total_patterns;
    double elapsed_seconds;
    
    /* Efficiency metrics */
    double patterns_per_input;
    double nodes_per_input;
    double learning_rate;  /* New patterns per 1000 inputs */
    
} Checkpoint;

void save_checkpoint(FILE *csv, Checkpoint *cp) {
    fprintf(csv, "%llu,%llu,%llu,%d,%.2f,%.6f,%.3f,%.3f\n",
            (unsigned long long)cp->input_count,
            (unsigned long long)cp->total_nodes,
            (unsigned long long)cp->total_edges,
            cp->total_patterns,
            cp->elapsed_seconds,
            cp->patterns_per_input,
            cp->nodes_per_input,
            cp->learning_rate);
    fflush(csv);
}

int count_patterns(Graph *g) {
    int count = 0;
    for (uint64_t i = 840; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) count++;
    }
    return count;
}

int main(int argc, char *argv[]) {
    /* Parse target input count */
    uint64_t target_inputs = 10000;  /* Default: 10K inputs (~minutes) */
    if (argc > 1) {
        target_inputs = strtoull(argv[1], NULL, 10);
    }
    
    printf("==============================================\n");
    printf("EXPERIMENT 4: LONG-TERM LEARNING EFFICIENCY\n");
    printf("==============================================\n\n");
    
    printf("Target inputs: %llu\n", (unsigned long long)target_inputs);
    
    if (target_inputs >= 100000) {
        printf("⚠️  This will take HOURS to complete\n");
    } else if (target_inputs >= 10000) {
        printf("⏱  This will take ~30-60 minutes\n");
    }
    
    printf("\nTesting: Does efficiency improve over time?\n");
    printf("Expected: Learning rate DECREASES as reuse INCREASES\n\n");
    
    /* Create brain with room to grow */
    const char *brain_path = "/tmp/longterm_brain.m";
    remove(brain_path);
    
    if (melvin_create_v2(brain_path, 10000, 50000, 8192, 0) != 0) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    Graph *g = melvin_open(brain_path, 10000, 50000, 8192);
    if (!g) {
        fprintf(stderr, "Failed to open brain\n");
        return 1;
    }
    
    /* CSV output */
    FILE *csv = fopen("benchmarks/data/experiment4_results.csv", "w");
    if (csv) {
        fprintf(csv, "inputs,nodes,edges,patterns,time_sec,patterns_per_input,nodes_per_input,learning_rate\n");
    }
    
    /* Count corpus size */
    int corpus_size = 0;
    while (CORPUS[corpus_size] != NULL) corpus_size++;
    
    printf("Corpus: %d entries\n", corpus_size);
    printf("Starting learning...\n\n");
    
    struct timespec start_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    uint64_t inputs = 0;
    uint64_t checkpoint_interval = 1000;  /* Report every 1K inputs */
    uint64_t next_checkpoint = checkpoint_interval;
    
    int prev_patterns = 0;
    uint64_t prev_checkpoint_inputs = 0;
    
    printf("Progress:\n");
    printf("%-10s %-10s %-10s %-12s %-12s %-10s\n",
           "Inputs", "Patterns", "Nodes", "P/Input", "Learn Rate", "Time");
    printf("------------------------------------------------------------------------\n");
    
    uint64_t progress_update = 100;  /* Update progress bar every 100 inputs */
    uint64_t next_progress = progress_update;
    
    while (inputs < target_inputs) {
        /* Select from corpus (cycle through) */
        const char *text = CORPUS[inputs % corpus_size];
        
        /* Feed each character */
        for (int i = 0; text[i] != '\0'; i++) {
            melvin_feed_byte(g, 0, (uint8_t)text[i], 1.0f);
            inputs++;
            
            /* Progress bar update? */
            if (inputs >= next_progress && inputs < target_inputs) {
                clock_gettime(CLOCK_MONOTONIC, &current_time);
                double elapsed = (current_time.tv_sec - start_time.tv_sec) +
                                (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
                
                float percent = (float)inputs / (float)target_inputs * 100.0f;
                
                /* Estimate time remaining */
                double rate = (double)inputs / elapsed;
                double remaining = (target_inputs - inputs) / rate;
                
                /* Progress bar [=========>          ] */
                int bar_width = 30;
                int filled = (int)(percent / 100.0f * bar_width);
                
                printf("\r[");
                for (int b = 0; b < bar_width; b++) {
                    if (b < filled) printf("=");
                    else if (b == filled) printf(">");
                    else printf(" ");
                }
                printf("] %.1f%% (%llu/%llu) ETA: %.0fs   ",
                       percent,
                       (unsigned long long)inputs,
                       (unsigned long long)target_inputs,
                       remaining);
                fflush(stdout);
                
                next_progress += progress_update;
            }
            
            /* Checkpoint? */
            if (inputs >= next_checkpoint || inputs >= target_inputs) {
                clock_gettime(CLOCK_MONOTONIC, &current_time);
                double elapsed = (current_time.tv_sec - start_time.tv_sec) +
                                (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
                
                Checkpoint cp;
                cp.input_count = inputs;
                cp.total_nodes = g->node_count;
                cp.total_edges = g->edge_count;
                cp.total_patterns = count_patterns(g);
                cp.elapsed_seconds = elapsed;
                cp.patterns_per_input = (double)cp.total_patterns / (double)inputs;
                cp.nodes_per_input = (double)cp.total_nodes / (double)inputs;
                
                /* Learning rate: new patterns per 1000 inputs */
                uint64_t input_delta = inputs - prev_checkpoint_inputs;
                int pattern_delta = cp.total_patterns - prev_patterns;
                cp.learning_rate = (double)pattern_delta / ((double)input_delta / 1000.0);
                
                /* Clear progress bar line */
                printf("\r%80s\r", "");
                
                printf("%-10llu %-10d %-10llu %-12.6f %-12.3f %.1fs\n",
                       (unsigned long long)inputs,
                       cp.total_patterns,
                       (unsigned long long)cp.total_nodes,
                       cp.patterns_per_input,
                       cp.learning_rate,
                       elapsed);
                
                if (csv) save_checkpoint(csv, &cp);
                
                prev_patterns = cp.total_patterns;
                prev_checkpoint_inputs = inputs;
                next_checkpoint += checkpoint_interval;
            }
        }
        
        /* Add space separator */
        melvin_feed_byte(g, 0, ' ', 0.5f);
        inputs++;
    }
    
    /* Clear progress bar */
    printf("\r%80s\r", "");
    
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    double total_time = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    /* Final analysis */
    printf("\n==============================================\n");
    printf("LONG-TERM LEARNING ANALYSIS\n");
    printf("==============================================\n\n");
    
    int final_patterns = count_patterns(g);
    
    printf("Final State:\n");
    printf("  Inputs:      %llu\n", (unsigned long long)inputs);
    printf("  Patterns:    %d\n", final_patterns);
    printf("  Nodes:       %llu\n", (unsigned long long)g->node_count);
    printf("  Edges:       %llu\n", (unsigned long long)g->edge_count);
    printf("  Time:        %.1f seconds (%.1f min)\n\n", total_time, total_time/60.0);
    
    printf("Efficiency Metrics:\n");
    printf("  Patterns/input:     %.6f\n", (double)final_patterns / (double)inputs);
    printf("  Inputs/second:      %.1f\n", (double)inputs / total_time);
    printf("  Patterns/second:    %.2f\n", (double)final_patterns / total_time);
    
    printf("\nKey Insight:\n");
    printf("  If learning rate DECREASED over time → Pattern reuse working!\n");
    printf("  If learning rate FLAT → No reuse benefit\n");
    printf("  Check CSV for trend analysis\n\n");
    
    if (csv) fclose(csv);
    melvin_close(g);
    
    printf("==============================================\n");
    printf("Results saved to: benchmarks/data/experiment4_results.csv\n");
    printf("==============================================\n\n");
    
    printf("To analyze trend: python3 benchmarks/analysis/plot_learning_curve.py\n");
    
    return 0;
}

