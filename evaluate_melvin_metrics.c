/*
 * evaluate_melvin_metrics.c - Hard metrics-based evaluation for Melvin
 * 
 * NO INTERPRETIVE SCORING - Only raw numeric metrics logged to CSV
 * 
 * Tests:
 * 1. Pattern Stability & Compression
 * 2. Locality of Activation
 * 3. Reaction to Surprise
 * 4. Memory Recall Under Load
 * 5. EXEC Function Triggering
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include "melvin.h"

/* Metrics structure - all numeric, no interpretation */
typedef struct {
    uint64_t step;
    float total_energy;
    uint32_t active_count;
    uint32_t active_groups;
    uint32_t pattern_nodes_active;
    uint32_t exec_fires;
    uint64_t node_count;
    uint64_t edge_count;
    float avg_activation;
    float avg_chaos;
} Metrics;

/* Count active inhibition groups (efficient - only check active nodes) */
static uint32_t count_active_groups(Graph *g) {
    if (!g || g->node_count == 0) return 0;
    
    uint32_t group_set[1000] = {0};  /* Track unique groups */
    uint32_t group_count = 0;
    
    /* Only check nodes with energy > threshold */
    uint32_t checked = 0;
    for (uint64_t i = 0; i < g->node_count && checked < 1000; i++) {
        if (g->nodes[i].energy > 0.01f) {  /* ACTIVE_THRESHOLD */
            uint32_t gid = g->nodes[i].inhib_group;
            if (gid < 1000) {
                /* Check if group already counted */
                int found = 0;
                for (uint32_t j = 0; j < group_count; j++) {
                    if (group_set[j] == gid) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    group_set[group_count++] = gid;
                }
            }
            checked++;
        }
    }
    
    return group_count;
}

/* Count active pattern nodes */
static uint32_t count_active_patterns(Graph *g) {
    if (!g || g->node_count == 0) return 0;
    
    uint32_t count = 0;
    uint32_t checked = 0;
    for (uint64_t i = 0; i < g->node_count && checked < 1000; i++) {
        if (g->nodes[i].type == NODE_TYPE_PATTERN && g->nodes[i].energy > 0.01f) {
            count++;
        }
        if (g->nodes[i].energy > 0.01f) checked++;
    }
    return count;
}

/* Count EXEC fires (track via exec_count increment) */
static uint32_t get_exec_fires(Graph *g) {
    if (!g || g->node_count == 0) return 0;
    
    uint32_t total_fires = 0;
    uint32_t checked = 0;
    for (uint64_t i = 0; i < g->node_count && checked < 1000; i++) {
        if (g->nodes[i].type == NODE_TYPE_EXEC) {
            total_fires += g->nodes[i].exec_count;
        }
        if (g->nodes[i].type == NODE_TYPE_EXEC) checked++;
    }
    return total_fires;
}

/* Log metrics to CSV */
static void log_metrics(FILE *csv, Metrics *m) {
    fprintf(csv, "%llu,%.6f,%u,%u,%u,%u,%llu,%llu,%.6f,%.6f\n",
            (unsigned long long)m->step,
            m->total_energy,
            m->active_count,
            m->active_groups,
            m->pattern_nodes_active,
            m->exec_fires,
            (unsigned long long)m->node_count,
            (unsigned long long)m->edge_count,
            m->avg_activation,
            m->avg_chaos);
}

/* Test 1: Pattern Stability & Compression */
static void test_pattern_stability(Graph *g, const char *csv_path) {
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open CSV: %s\n", csv_path);
        return;
    }
    
    /* CSV header */
    fprintf(csv, "step,total_energy,active_count,active_groups,pattern_nodes_active,exec_fires,node_count,edge_count,avg_activation,avg_chaos\n");
    
    const char *pattern = "ABABABABABABABABABAB";
    uint64_t step = 0;
    
    /* Initial state */
    Metrics m = {0};
    m.step = step++;
    m.total_energy = g->total_energy;
    m.active_count = g->active_count;
    m.active_groups = count_active_groups(g);
    m.pattern_nodes_active = count_active_patterns(g);
    m.exec_fires = get_exec_fires(g);
    m.node_count = g->node_count;
    m.edge_count = g->edge_count;
    m.avg_activation = g->avg_activation;
    m.avg_chaos = g->avg_chaos;
    log_metrics(csv, &m);
    
    /* Feed pattern */
    for (int i = 0; pattern[i]; i++) {
        melvin_feed_byte(g, 0, pattern[i], 0.2f);
        melvin_run_physics(g);
        
        m.step = step++;
        m.total_energy = g->total_energy;
        m.active_count = g->active_count;
        m.active_groups = count_active_groups(g);
        m.pattern_nodes_active = count_active_patterns(g);
        m.exec_fires = get_exec_fires(g);
        m.node_count = g->node_count;
        m.edge_count = g->edge_count;
        m.avg_activation = g->avg_activation;
        m.avg_chaos = g->avg_chaos;
        log_metrics(csv, &m);
    }
    
    /* Post-pattern: run physics a few more times to see if pattern stabilizes */
    for (int i = 0; i < 10; i++) {
        melvin_run_physics(g);
        m.step = step++;
        m.total_energy = g->total_energy;
        m.active_count = g->active_count;
        m.active_groups = count_active_groups(g);
        m.pattern_nodes_active = count_active_patterns(g);
        m.exec_fires = get_exec_fires(g);
        m.node_count = g->node_count;
        m.edge_count = g->edge_count;
        m.avg_activation = g->avg_activation;
        m.avg_chaos = g->avg_chaos;
        log_metrics(csv, &m);
    }
    
    fclose(csv);
}

/* Test 2: Locality of Activation */
static void test_locality(Graph *g, const char *csv_path) {
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open CSV: %s\n", csv_path);
        return;
    }
    
    fprintf(csv, "step,total_energy,active_count,active_groups,pattern_nodes_active,exec_fires,node_count,edge_count,avg_activation,avg_chaos\n");
    
    const char *input = "HelloWorldHelloWorldHelloWorld";
    uint64_t step = 0;
    
    Metrics m = {0};
    
    for (int i = 0; input[i]; i++) {
        melvin_feed_byte(g, 0, input[i], 0.2f);
        melvin_run_physics(g);
        
        m.step = step++;
        m.total_energy = g->total_energy;
        m.active_count = g->active_count;
        m.active_groups = count_active_groups(g);
        m.pattern_nodes_active = count_active_patterns(g);
        m.exec_fires = get_exec_fires(g);
        m.node_count = g->node_count;
        m.edge_count = g->edge_count;
        m.avg_activation = g->avg_activation;
        m.avg_chaos = g->avg_chaos;
        log_metrics(csv, &m);
    }
    
    fclose(csv);
}

/* Test 3: Reaction to Surprise */
static void test_surprise(Graph *g, const char *csv_path) {
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open CSV: %s\n", csv_path);
        return;
    }
    
    fprintf(csv, "step,total_energy,active_count,active_groups,pattern_nodes_active,exec_fires,node_count,edge_count,avg_activation,avg_chaos\n");
    
    const char *normal = "1010101010101010";
    const char *anomaly = "1010101011101010";
    uint64_t step = 0;
    
    Metrics m = {0};
    
    /* Feed normal sequence */
    for (int i = 0; normal[i]; i++) {
        melvin_feed_byte(g, 0, normal[i], 0.2f);
        melvin_run_physics(g);
        
        m.step = step++;
        m.total_energy = g->total_energy;
        m.active_count = g->active_count;
        m.active_groups = count_active_groups(g);
        m.pattern_nodes_active = count_active_patterns(g);
        m.exec_fires = get_exec_fires(g);
        m.node_count = g->node_count;
        m.edge_count = g->edge_count;
        m.avg_activation = g->avg_activation;
        m.avg_chaos = g->avg_chaos;
        log_metrics(csv, &m);
    }
    
    /* Feed anomaly */
    for (int i = 0; anomaly[i]; i++) {
        melvin_feed_byte(g, 0, anomaly[i], 0.2f);
        melvin_run_physics(g);
        
        m.step = step++;
        m.total_energy = g->total_energy;
        m.active_count = g->active_count;
        m.active_groups = count_active_groups(g);
        m.pattern_nodes_active = count_active_patterns(g);
        m.exec_fires = get_exec_fires(g);
        m.node_count = g->node_count;
        m.edge_count = g->edge_count;
        m.avg_activation = g->avg_activation;
        m.avg_chaos = g->avg_chaos;
        log_metrics(csv, &m);
    }
    
    fclose(csv);
}

/* Test 4: Memory Recall Under Load */
static void test_memory_recall(Graph *g, const char *csv_path) {
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open CSV: %s\n", csv_path);
        return;
    }
    
    fprintf(csv, "step,total_energy,active_count,active_groups,pattern_nodes_active,exec_fires,node_count,edge_count,avg_activation,avg_chaos\n");
    
    const char *pattern = "MSG:START";
    uint64_t step = 0;
    
    Metrics m = {0};
    
    /* Feed 1000 random bytes */
    srand(time(NULL));
    for (int i = 0; i < 1000; i++) {
        uint8_t b = rand() % 256;
        melvin_feed_byte(g, 0, b, 0.1f);
        if (i % 100 == 0) {
            melvin_run_physics(g);
            m.step = step++;
            m.total_energy = g->total_energy;
            m.active_count = g->active_count;
            m.active_groups = count_active_groups(g);
            m.pattern_nodes_active = count_active_patterns(g);
            m.exec_fires = get_exec_fires(g);
            m.node_count = g->node_count;
            m.edge_count = g->edge_count;
            m.avg_activation = g->avg_activation;
            m.avg_chaos = g->avg_chaos;
            log_metrics(csv, &m);
        }
    }
    
    melvin_run_physics(g);
    
    /* Search for pattern */
    for (int i = 0; pattern[i]; i++) {
        melvin_feed_byte(g, 0, pattern[i], 0.3f);
        melvin_run_physics(g);
        
        m.step = step++;
        m.total_energy = g->total_energy;
        m.active_count = g->active_count;
        m.active_groups = count_active_groups(g);
        m.pattern_nodes_active = count_active_patterns(g);
        m.exec_fires = get_exec_fires(g);
        m.node_count = g->node_count;
        m.edge_count = g->edge_count;
        m.avg_activation = g->avg_activation;
        m.avg_chaos = g->avg_chaos;
        log_metrics(csv, &m);
    }
    
    fclose(csv);
}

/* Test 5: EXEC Function Triggering */
static void test_exec_triggering(Graph *g, const char *csv_path) {
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "ERROR: Cannot open CSV: %s\n", csv_path);
        return;
    }
    
    fprintf(csv, "step,total_energy,active_count,active_groups,pattern_nodes_active,exec_fires,node_count,edge_count,avg_activation,avg_chaos\n");
    
    const char *pattern = "RUN(3,5)";
    uint64_t step = 0;
    uint32_t exec_fires_before = get_exec_fires(g);
    
    Metrics m = {0};
    
    /* Feed pattern */
    for (int i = 0; pattern[i]; i++) {
        melvin_feed_byte(g, 0, pattern[i], 0.3f);
        melvin_run_physics(g);
        
        m.step = step++;
        m.total_energy = g->total_energy;
        m.active_count = g->active_count;
        m.active_groups = count_active_groups(g);
        m.pattern_nodes_active = count_active_patterns(g);
        m.exec_fires = get_exec_fires(g);
        m.node_count = g->node_count;
        m.edge_count = g->edge_count;
        m.avg_activation = g->avg_activation;
        m.avg_chaos = g->avg_chaos;
        log_metrics(csv, &m);
    }
    
    uint32_t exec_fires_after = get_exec_fires(g);
    
    fclose(csv);
    
    /* Log EXEC firing delta to separate file */
    char exec_log_path[512];
    snprintf(exec_log_path, sizeof(exec_log_path), "%s.exec_fires", csv_path);
    FILE *exec_log = fopen(exec_log_path, "w");
    if (exec_log) {
        fprintf(exec_log, "exec_fires_before,exec_fires_after,delta\n");
        fprintf(exec_log, "%u,%u,%u\n", exec_fires_before, exec_fires_after, 
                exec_fires_after - exec_fires_before);
        fclose(exec_log);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <brain.m> [test_number]\n", argv[0]);
        fprintf(stderr, "  test_number: 1-5 (or omit to run all)\n");
        return 1;
    }
    
    const char *brain_path = argv[1];
    int test_num = (argc > 2) ? atoi(argv[2]) : 0;
    
    Graph *g = melvin_open(brain_path, 10000, 50000, 1048576);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to open brain: %s\n", brain_path);
        return 1;
    }
    
    char csv_path[512];
    const char *results_dir = "evaluation_results";
    
    /* Create results directory */
    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", results_dir);
    system(mkdir_cmd);
    
    if (test_num == 0 || test_num == 1) {
        snprintf(csv_path, sizeof(csv_path), "%s/test_1_pattern_stability.csv", results_dir);
        test_pattern_stability(g, csv_path);
        printf("Test 1 complete: %s\n", csv_path);
    }
    
    if (test_num == 0 || test_num == 2) {
        snprintf(csv_path, sizeof(csv_path), "%s/test_2_locality.csv", results_dir);
        test_locality(g, csv_path);
        printf("Test 2 complete: %s\n", csv_path);
    }
    
    if (test_num == 0 || test_num == 3) {
        snprintf(csv_path, sizeof(csv_path), "%s/test_3_surprise.csv", results_dir);
        test_surprise(g, csv_path);
        printf("Test 3 complete: %s\n", csv_path);
    }
    
    if (test_num == 0 || test_num == 4) {
        snprintf(csv_path, sizeof(csv_path), "%s/test_4_memory_recall.csv", results_dir);
        test_memory_recall(g, csv_path);
        printf("Test 4 complete: %s\n", csv_path);
    }
    
    if (test_num == 0 || test_num == 5) {
        snprintf(csv_path, sizeof(csv_path), "%s/test_5_exec_triggering.csv", results_dir);
        test_exec_triggering(g, csv_path);
        printf("Test 5 complete: %s\n", csv_path);
    }
    
    melvin_close(g);
    return 0;
}

