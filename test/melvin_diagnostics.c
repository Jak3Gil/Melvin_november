/*
 * MELVIN DIAGNOSTICS - Implementation
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include "melvin_diagnostics.h"

// Global diagnostic state
bool g_diagnostics_enabled = false;
FILE *g_node_diag_log = NULL;
FILE *g_edge_diag_log = NULL;
FILE *g_global_diag_log = NULL;

static uint64_t g_event_counter = 0;

void diagnostics_init(const char *output_dir) {
    if (!output_dir) {
        output_dir = ".";
    }
    
    char node_path[512];
    char edge_path[512];
    char global_path[512];
    
    snprintf(node_path, sizeof(node_path), "%s/node_diagnostics.csv", output_dir);
    snprintf(edge_path, sizeof(edge_path), "%s/edge_diagnostics.csv", output_dir);
    snprintf(global_path, sizeof(global_path), "%s/global_diagnostics.csv", output_dir);
    
    g_node_diag_log = fopen(node_path, "w");
    g_edge_diag_log = fopen(edge_path, "w");
    g_global_diag_log = fopen(global_path, "w");
    
    if (g_node_diag_log) {
        fprintf(g_node_diag_log, "event_index,node_id,state_before,state_after,prediction_before,prediction_after,prediction_error,fe_inst,fe_ema,traffic_ema\n");
    }
    
    if (g_edge_diag_log) {
        fprintf(g_edge_diag_log, "event_index,src_id,dst_id,weight_before,weight_after,delta_w,eligibility,usage,last_energy\n");
    }
    
    if (g_global_diag_log) {
        fprintf(g_global_diag_log, "event_index,mean_state,var_state,mean_prediction_error,var_prediction_error,mean_fe_ema,var_fe_ema,mean_weight,var_weight,frac_strong_edges,num_pattern_nodes,num_seq_edges,num_chan_edges,num_bonds\n");
    }
    
    g_diagnostics_enabled = true;
    g_event_counter = 0;
}

void diagnostics_cleanup(void) {
    if (g_node_diag_log) {
        fclose(g_node_diag_log);
        g_node_diag_log = NULL;
    }
    
    if (g_edge_diag_log) {
        fclose(g_edge_diag_log);
        g_edge_diag_log = NULL;
    }
    
    if (g_global_diag_log) {
        fclose(g_global_diag_log);
        g_global_diag_log = NULL;
    }
    
    g_diagnostics_enabled = false;
}

void diagnostics_log_node_update(
    uint64_t event_index,
    uint64_t node_id,
    float state_before,
    float state_after,
    float prediction_before,
    float prediction_after,
    float prediction_error,
    float fe_inst,
    float fe_ema,
    float traffic_ema
) {
    if (!g_diagnostics_enabled || !g_node_diag_log) return;
    
    fprintf(g_node_diag_log, "%llu,%llu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            (unsigned long long)event_index,
            (unsigned long long)node_id,
            state_before,
            state_after,
            prediction_before,
            prediction_after,
            prediction_error,
            fe_inst,
            fe_ema,
            traffic_ema);
    fflush(g_node_diag_log);
}

void diagnostics_log_edge_update(
    uint64_t event_index,
    uint64_t src_id,
    uint64_t dst_id,
    float weight_before,
    float weight_after,
    float delta_w,
    float eligibility,
    float usage,
    float last_energy
) {
    if (!g_diagnostics_enabled || !g_edge_diag_log) return;
    
    fprintf(g_edge_diag_log, "%llu,%llu,%llu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            (unsigned long long)event_index,
            (unsigned long long)src_id,
            (unsigned long long)dst_id,
            weight_before,
            weight_after,
            delta_w,
            eligibility,
            usage,
            last_energy);
    fflush(g_edge_diag_log);
}

void diagnostics_log_global_snapshot(
    uint64_t event_index,
    float mean_state,
    float var_state,
    float mean_prediction_error,
    float var_prediction_error,
    float mean_fe_ema,
    float var_fe_ema,
    float mean_weight,
    float var_weight,
    float frac_strong_edges,
    uint64_t num_pattern_nodes,
    uint64_t num_seq_edges,
    uint64_t num_chan_edges,
    uint64_t num_bonds
) {
    if (!g_diagnostics_enabled || !g_global_diag_log) return;
    
    fprintf(g_global_diag_log, "%llu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu\n",
            (unsigned long long)event_index,
            mean_state,
            var_state,
            mean_prediction_error,
            var_prediction_error,
            mean_fe_ema,
            var_fe_ema,
            mean_weight,
            var_weight,
            frac_strong_edges,
            (unsigned long long)num_pattern_nodes,
            (unsigned long long)num_seq_edges,
            (unsigned long long)num_chan_edges,
            (unsigned long long)num_bonds);
    fflush(g_global_diag_log);
}

// Helper to get current event counter
uint64_t diagnostics_get_event_counter(void) {
    return g_event_counter;
}

void diagnostics_increment_event_counter(void) {
    g_event_counter++;
}

