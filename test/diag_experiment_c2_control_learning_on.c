/*
 * DIAGNOSTIC EXPERIMENT C.2: Control Learning with Learning ON
 * 
 * Purpose: Verify control learning actually improves with learning enabled
 * 
 * Same 1D env (position x, LEFT/RIGHT/STAY, +1 at target)
 * Learning ON - verify reward injection and weight updates
 * 
 * Log:
 * - Episode returns vs episode index
 * - STATE→ACTION edge weights over time
 * 
 * Goal:
 * - Reward curve above random baseline (even modest improvement)
 * - Some edges drifting to consistent "go toward target" behavior
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "melvin.c"
#include "melvin_diagnostics.c"

#define TEST_FILE "diag_c2_control.m"
#define CHANNEL_STATE 10
#define CHANNEL_ACTION 20
#define CHANNEL_REWARD 30
#define STATE_MIN 0
#define STATE_MAX 10
#define TARGET_STATE 5
#define MAX_EPISODES 50
#define MAX_STEPS_PER_EPISODE 15
#define EVENTS_PER_STEP 10

// Simple 1D environment
typedef struct {
    int state_x;
    float last_reward;
} Environment;

static float compute_reward(int state_x) {
    return (state_x == TARGET_STATE) ? 1.0f : 0.0f;
}

static int apply_action(int state_x, uint8_t action) {
    int new_state = state_x;
    if (action == 0 && new_state > STATE_MIN) {
        new_state--;  // LEFT
    } else if (action == 2 && new_state < STATE_MAX) {
        new_state++;  // RIGHT
    }
    // action == 1 is STAY
    return new_state;
}

static uint8_t read_action_from_melvin(MelvinRuntime *rt) {
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
    for (uint8_t action = 0; action < 3; action++) {
        uint64_t action_node_id = (uint64_t)action + 3000000ULL;
        uint64_t action_idx = find_node_index_by_id(rt->file, action_node_id);
        
        if (action_idx == UINT64_MAX) {
            ingest_byte(rt, CHANNEL_ACTION, 200 + action, 0.1f);
            melvin_process_n_events(rt, 5);
        }
    }
}

// Track key edges: state nodes -> action nodes
typedef struct {
    uint64_t state_node_id;
    uint64_t action_node_id;
    float weight;
    int found;
} TrackedEdge;

static int find_tracked_edges(MelvinFile *file, TrackedEdge *edges, int max_edges) {
    int count = 0;
    GraphHeaderDisk *gh = file->graph_header;
    
    // Find state nodes (channel 10, byte values 0-255)
    // Find action nodes (3000000-3000002)
    // Find edges between them
    
    for (uint64_t e = 0; e < gh->num_edges && count < max_edges; e++) {
        EdgeDisk *edge = &file->edges[e];
        if (edge->src == UINT64_MAX) continue;
        
        // Check if this is a state->action edge
        uint64_t src = edge->src;
        uint64_t dst = edge->dst;
        
        // State nodes: 1000000-1000255 (DATA nodes for bytes 0-255)
        // Action nodes: 3000000-3000002
        if (src >= 1000000ULL && src <= 1000255ULL && 
            dst >= 3000000ULL && dst <= 3000002ULL) {
            edges[count].state_node_id = src;
            edges[count].action_node_id = dst;
            edges[count].weight = edge->weight;
            edges[count].found = 1;
            count++;
        }
    }
    
    return count;
}

int main() {
    printf("========================================\n");
    printf("DIAGNOSTIC EXPERIMENT C.2: Control Learning ON\n");
    printf("========================================\n\n");
    
    diagnostics_init("diag_c2_results");
    
    srand(time(NULL));
    
    // Create test file with learning enabled
    GraphParams params;
    params.decay_rate = 0.95f;
    params.reward_lambda = 0.2f;  // Higher reward strength
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.02f;  // Learning enabled
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
    TrackedEdge tracked_edges[20];
    int num_tracked = 0;
    
    printf("Running %d episodes with learning enabled...\n\n", MAX_EPISODES);
    
    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        // Reset environment
        env.state_x = STATE_MIN + (rand() % (STATE_MAX - STATE_MIN + 1));
        env.last_reward = 0.0f;
        
        float episode_return = 0.0f;
        
        for (int step = 0; step < MAX_STEPS_PER_EPISODE; step++) {
            diagnostics_increment_event_counter();
            
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
            
            // 5. Inject reward for learning (into state node that led to action)
            // Find the state node that was just activated
            uint64_t state_node_id = (uint64_t)((env.state_x * 255) / STATE_MAX) + 1000000ULL;
            inject_reward(&rt, state_node_id, reward);
            melvin_process_n_events(&rt, EVENTS_PER_STEP);
            
            // Update environment
            env.state_x = new_state;
            env.last_reward = reward;
            
            // Check if reached target
            if (env.state_x == TARGET_STATE) {
                break;
            }
        }
        
        episode_returns[episode] = episode_return;
        
        // Track edges periodically
        if (episode % 10 == 0 || episode == MAX_EPISODES - 1) {
            num_tracked = find_tracked_edges(&file, tracked_edges, 20);
            
            printf("Episode %d: return=%.1f", episode, episode_return);
            if (num_tracked > 0) {
                printf(", tracked edges=%d", num_tracked);
                // Show first few edge weights
                for (int i = 0; i < num_tracked && i < 3; i++) {
                    printf(", w[%llu->%llu]=%.3f",
                           (unsigned long long)tracked_edges[i].state_node_id % 1000000,
                           (unsigned long long)tracked_edges[i].action_node_id % 1000000,
                           tracked_edges[i].weight);
                }
            }
            printf("\n");
        }
    }
    
    // Analysis
    printf("\n========================================\n");
    printf("ANALYSIS:\n");
    printf("========================================\n");
    
    // Compute average return for first and last 10 episodes
    float avg_first = 0.0f;
    float avg_last = 0.0f;
    float avg_random = 0.0f;  // Random baseline (1/11 chance of hitting target per step)
    
    for (int i = 0; i < 10 && i < MAX_EPISODES; i++) {
        avg_first += episode_returns[i];
    }
    avg_first /= 10.0f;
    
    int last_start = (MAX_EPISODES > 10) ? (MAX_EPISODES - 10) : 0;
    for (int i = last_start; i < MAX_EPISODES; i++) {
        avg_last += episode_returns[i];
    }
    avg_last /= (float)(MAX_EPISODES - last_start);
    
    // Random baseline: probability of hitting target in 15 steps
    // With random actions, roughly 1/11 chance per step, so ~1.36 expected return per episode
    avg_random = 1.36f;
    
    printf("Average return (first 10): %.3f\n", avg_first);
    printf("Average return (last 10): %.3f\n", avg_last);
    printf("Random baseline: %.3f\n", avg_random);
    printf("Improvement: %.3f\n", avg_last - avg_first);
    printf("\n");
    
    // Check tracked edges
    num_tracked = find_tracked_edges(&file, tracked_edges, 20);
    printf("Tracked STATE→ACTION edges: %d\n", num_tracked);
    if (num_tracked > 0) {
        printf("Sample edge weights:\n");
        for (int i = 0; i < num_tracked && i < 5; i++) {
            printf("  State %llu → Action %llu: weight=%.6f\n",
                   (unsigned long long)(tracked_edges[i].state_node_id % 1000000),
                   (unsigned long long)(tracked_edges[i].action_node_id % 1000000),
                   tracked_edges[i].weight);
        }
    }
    printf("\n");
    
    // Success criteria
    int passed = 1;
    
    if (avg_last <= avg_random) {
        printf("❌ FAIL: Return not above random baseline\n");
        passed = 0;
    } else {
        printf("✓ PASS: Return above random baseline\n");
    }
    
    if (avg_last <= avg_first) {
        printf("❌ FAIL: No improvement over episodes\n");
        passed = 0;
    } else {
        printf("✓ PASS: Improvement detected (%.3f)\n", avg_last - avg_first);
    }
    
    if (num_tracked == 0) {
        printf("⚠ WARNING: No STATE→ACTION edges found\n");
    } else {
        printf("✓ PASS: STATE→ACTION edges exist\n");
    }
    
    // Log episode returns to CSV
    FILE *returns_log = fopen("diag_c2_results/episode_returns.csv", "w");
    if (returns_log) {
        fprintf(returns_log, "episode,return\n");
        for (int i = 0; i < MAX_EPISODES; i++) {
            fprintf(returns_log, "%d,%.3f\n", i, episode_returns[i]);
        }
        fclose(returns_log);
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    diagnostics_cleanup();
    
    if (passed) {
        printf("\n✓ PASS: Control learning works in principle\n");
        printf("Results logged to diag_c2_results/\n");
        return 0;
    } else {
        printf("\n❌ FAIL: Control learning needs improvement\n");
        return 1;
    }
}

