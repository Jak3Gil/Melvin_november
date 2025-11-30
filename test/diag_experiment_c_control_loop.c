/*
 * DIAGNOSTIC EXPERIMENT C: Control Loop Concept Check
 * 
 * Purpose: Verify control wiring is correct and isolate segfault
 * 
 * Setup:
 * - Simplified 1D environment: state x in [0, N], actions {LEFT, RIGHT, STAY}
 * - Reward: +1 when x == target, else 0
 * 
 * First run with learning disabled to check for segfaults
 * Then run with learning enabled to check weight changes
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>
#include <setjmp.h>
#include "melvin.c"
#include "melvin_diagnostics.c"

#define TEST_FILE "diag_c_control.m"
#define CHANNEL_STATE 10
#define CHANNEL_ACTION 20
#define CHANNEL_REWARD 30
#define STATE_MIN 0
#define STATE_MAX 10
#define TARGET_STATE 5
#define MAX_EPISODES 20
#define MAX_STEPS_PER_EPISODE 10
#define EVENTS_PER_STEP 5

static jmp_buf segfault_handler;

static void segfault_signal_handler(int sig) {
    fprintf(stderr, "\nSEGFAULT DETECTED!\n");
    longjmp(segfault_handler, 1);
}

// Simple 1D environment
typedef struct {
    int state_x;
    float last_reward;
} Environment;

static float compute_reward(int state_x) {
    return (state_x == TARGET_STATE) ? 1.0f : 0.0f;
}

static int apply_action(int state_x, uint8_t action) {
    // Action 0 = left, 1 = stay, 2 = right
    int new_state = state_x;
    if (action == 0 && new_state > STATE_MIN) {
        new_state--;
    } else if (action == 2 && new_state < STATE_MAX) {
        new_state++;
    }
    return new_state;
}

static uint8_t read_action_from_melvin(MelvinRuntime *rt) {
    // Read activation of action nodes (0, 1, 2)
    // Action node with highest activation wins
    float max_activation = -1.0f;
    uint8_t chosen_action = 1;  // Default: stay
    
    for (uint8_t action = 0; action < 3; action++) {
        uint64_t action_node_id = (uint64_t)action + 3000000ULL;
        uint64_t action_idx = find_node_index_by_id(rt->file, action_node_id);
        
        if (action_idx != UINT64_MAX) {
            NodeDisk *node = &rt->file->nodes[action_idx];
            if (node->state > max_activation) {
                max_activation = node->state;
                chosen_action = action;
            }
        }
    }
    
    return chosen_action;
}

static void inject_state_and_reward(MelvinRuntime *rt, int state_x, float reward) {
    // Inject state as byte (0-10 mapped to 0-255)
    uint8_t state_byte = (uint8_t)((state_x * 255) / STATE_MAX);
    ingest_byte(rt, CHANNEL_STATE, state_byte, 1.0f);
    
    // Inject reward as byte (0.0-1.0 mapped to 0-255)
    uint8_t reward_byte = (uint8_t)(reward * 255.0f);
    ingest_byte(rt, CHANNEL_REWARD, reward_byte, 1.0f);
}

static void create_action_nodes(MelvinRuntime *rt) {
    // Create action nodes if they don't exist
    for (uint8_t action = 0; action < 3; action++) {
        uint64_t action_node_id = (uint64_t)action + 3000000ULL;
        uint64_t action_idx = find_node_index_by_id(rt->file, action_node_id);
        
        if (action_idx == UINT64_MAX) {
            // Create node by ingesting a unique byte
            ingest_byte(rt, CHANNEL_ACTION, 200 + action, 0.1f);
            melvin_process_n_events(rt, 5);
        }
    }
}

static int run_control_test(int enable_learning, const char *diag_dir) {
    printf("\n========================================\n");
    printf("CONTROL TEST: Learning %s\n", enable_learning ? "ENABLED" : "DISABLED");
    printf("========================================\n\n");
    
    diagnostics_init(diag_dir);
    
    // Set up segfault handler
    signal(SIGSEGV, segfault_signal_handler);
    signal(SIGBUS, segfault_signal_handler);
    
    if (setjmp(segfault_handler) != 0) {
        fprintf(stderr, "SEGFAULT occurred - aborting test\n");
        diagnostics_cleanup();
        return 1;
    }
    
    // Create test file
    GraphParams params;
    params.decay_rate = 0.95f;
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.exec_threshold = 0.75f;
    params.learning_rate = enable_learning ? 0.02f : 0.0f;  // Disable learning if requested
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create test file\n");
        diagnostics_cleanup();
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        diagnostics_cleanup();
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        diagnostics_cleanup();
        return 1;
    }
    
    // Create action nodes
    create_action_nodes(&rt);
    
    Environment env;
    float episode_returns[MAX_EPISODES];
    float initial_edge_weights[10];
    int num_tracked_edges = 0;
    
    // Track some key edges before learning
    GraphHeaderDisk *gh = file.graph_header;
    for (uint64_t e = 0; e < gh->num_edges && num_tracked_edges < 10; e++) {
        EdgeDisk *edge = &file.edges[e];
        if (edge->src != UINT64_MAX) {
            initial_edge_weights[num_tracked_edges] = edge->weight;
            num_tracked_edges++;
        }
    }
    
    printf("Running %d episodes...\n", MAX_EPISODES);
    
    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        // Reset environment
        env.state_x = STATE_MIN + (rand() % (STATE_MAX - STATE_MIN + 1));
        env.last_reward = 0.0f;
        
        float episode_return = 0.0f;
        
        for (int step = 0; step < MAX_STEPS_PER_EPISODE; step++) {
            diagnostics_increment_event_counter();
            uint64_t event_idx = diagnostics_get_event_counter();
            
            // 1. Environment sends state + reward
            inject_state_and_reward(&rt, env.state_x, env.last_reward);
            melvin_process_n_events(&rt, EVENTS_PER_STEP);
            
            // 2. Read action from Melvin
            uint8_t action = read_action_from_melvin(&rt);
            
            // 3. Apply action
            int new_state = apply_action(env.state_x, action);
            
            // 4. Compute reward
            float reward = compute_reward(new_state);
            episode_return += reward;
            
            // 5. Inject reward for learning
            if (enable_learning) {
                inject_reward(&rt, 0, reward);
            }
            melvin_process_n_events(&rt, EVENTS_PER_STEP);
            
            // Log interaction
            printf("  Episode %d, Step %d: x=%d, action=%d, new_x=%d, reward=%.1f\n",
                   episode, step, env.state_x, action, new_state, reward);
            
            // Update environment
            env.state_x = new_state;
            env.last_reward = reward;
            
            // Check if reached target
            if (env.state_x == TARGET_STATE) {
                break;
            }
        }
        
        episode_returns[episode] = episode_return;
        
        printf("Episode %d: return=%.1f\n", episode, episode_return);
    }
    
    // Check if weights changed
    printf("\nChecking weight changes...\n");
    int weights_changed = 0;
    int edge_idx = 0;
    for (uint64_t e = 0; e < gh->num_edges && edge_idx < num_tracked_edges; e++) {
        EdgeDisk *edge = &file.edges[e];
        if (edge->src != UINT64_MAX) {
            float weight_diff = fabsf(edge->weight - initial_edge_weights[edge_idx]);
            if (weight_diff > 0.001f) {
                weights_changed++;
                printf("  Edge %llu->%llu: weight changed from %.6f to %.6f (diff=%.6f)\n",
                       (unsigned long long)edge->src,
                       (unsigned long long)edge->dst,
                       initial_edge_weights[edge_idx],
                       edge->weight,
                       weight_diff);
            }
            edge_idx++;
        }
    }
    
    // Analysis
    printf("\n========================================\n");
    printf("RESULTS:\n");
    printf("========================================\n");
    
    float avg_return = 0.0f;
    for (int i = 0; i < MAX_EPISODES; i++) {
        avg_return += episode_returns[i];
    }
    avg_return /= MAX_EPISODES;
    
    printf("Average episode return: %.3f\n", avg_return);
    printf("Weights changed: %d/%d\n", weights_changed, num_tracked_edges);
    
    if (enable_learning) {
        if (weights_changed > 0) {
            printf("✓ Learning is active (weights changing)\n");
        } else {
            printf("❌ Learning not working (weights not changing)\n");
        }
    } else {
        if (weights_changed == 0) {
            printf("✓ Learning disabled correctly\n");
        } else {
            printf("⚠ WARNING: Weights changed even with learning disabled\n");
        }
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    diagnostics_cleanup();
    
    return 0;
}

int main() {
    printf("========================================\n");
    printf("DIAGNOSTIC EXPERIMENT C: Control Loop Check\n");
    printf("========================================\n");
    
    srand(time(NULL));
    
    // First: Run with learning disabled to check for segfaults
    printf("\nPhase 1: Testing without learning (checking for segfaults)...\n");
    int result1 = run_control_test(0, "diag_c_no_learning_results");
    
    if (result1 != 0) {
        printf("\n❌ FAIL: Segfault or error occurred without learning\n");
        return 1;
    }
    
    printf("\n✓ PASS: No segfault without learning\n");
    
    // Second: Run with learning enabled
    printf("\nPhase 2: Testing with learning enabled...\n");
    int result2 = run_control_test(1, "diag_c_with_learning_results");
    
    if (result2 != 0) {
        printf("\n❌ FAIL: Segfault or error occurred with learning\n");
        return 1;
    }
    
    printf("\n✓ PASS: Control loop test completed\n");
    printf("\nCheck diag_c_*_results/ for detailed diagnostics\n");
    
    return 0;
}

