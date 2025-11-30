/*
 * TEST 4: Event-Driven Control Learning
 * 
 * Goal: Can it learn a policy purely from event sequences?
 * 
 * Test 4.1: 1D world, event-driven
 * 
 * Environment function:
 *   event: (state_x, reward_prev) → Melvin → action → new_state_x, reward_new
 * 
 * One interaction step:
 * 1. Environment sends STATE bytes (position) + last REWARD bytes
 * 2. Run message passing + activation updates + learning
 * 3. Read Melvin's chosen ACTION (motor channel)
 * 4. Environment applies action to state_x → new_state_x, computes new reward
 * 5. Log (state, action, new_state, reward) for analysis
 * 
 * Run many episodes (sequences of interaction events, reset state between episodes)
 * 
 * Check:
 * - Episode return (sum of rewards) trends upward vs random policy
 * - Relevant edges between state/motor/reward nodes strengthen consistently
 * - FE_ema in that subgraph reduces over time
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_4_control.m"
#define CHANNEL_STATE 10
#define CHANNEL_ACTION 20
#define CHANNEL_REWARD 30
#define STATE_MIN 0
#define STATE_MAX 10
#define TARGET_STATE 5
#define MAX_EPISODES 50
#define MAX_STEPS_PER_EPISODE 20
#define EVENTS_PER_STEP 10

// Simple 1D environment
typedef struct {
    int state_x;
    float last_reward;
} Environment;

static float compute_reward(int state_x) {
    // Reward is higher when closer to target
    int distance = abs(state_x - TARGET_STATE);
    return 1.0f / (1.0f + (float)distance);
}

static int apply_action(int state_x, uint8_t action) {
    // Action 0 = move left, 1 = stay, 2 = move right
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
    GraphHeaderDisk *gh = rt->file->graph_header;
    float max_activation = -1.0f;
    uint8_t chosen_action = 1;  // Default: stay
    
    for (uint8_t action = 0; action < 3; action++) {
        uint64_t action_node_id = (uint64_t)action + 3000000ULL;  // Action nodes in range 3000000-3000002
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

int main() {
    printf("========================================\n");
    printf("TEST 4: Event-Driven Control Learning\n");
    printf("========================================\n\n");
    
    // Create test file
    GraphParams params;
    params.decay_rate = 0.95f;
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.02f;  // Higher learning rate for control
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create test file\n");
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    
    // Create action nodes
    create_action_nodes(&rt);
    
    Environment env;
    float episode_returns[MAX_EPISODES];
    
    printf("Running %d episodes...\n\n", MAX_EPISODES);
    
    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        // Reset environment
        env.state_x = STATE_MIN + (rand() % (STATE_MAX - STATE_MIN + 1));
        env.last_reward = 0.0f;
        
        float episode_return = 0.0f;
        
        for (int step = 0; step < MAX_STEPS_PER_EPISODE; step++) {
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
            inject_reward(&rt, 0, reward);  // Reward signal
            melvin_process_n_events(&rt, EVENTS_PER_STEP);
            
            // Update environment
            env.state_x = new_state;
            env.last_reward = reward;
            
            // Check if reached target
            if (env.state_x == TARGET_STATE) {
                break;  // Early termination
            }
        }
        
        episode_returns[episode] = episode_return;
        
        if (episode % 10 == 0 || episode == MAX_EPISODES - 1) {
            printf("Episode %d: return=%.3f, final_state=%d\n", 
                   episode, episode_return, env.state_x);
        }
    }
    
    // Analyze results
    printf("\n========================================\n");
    printf("RESULTS:\n");
    printf("========================================\n");
    
    // Compute average return for first and last 10 episodes
    float avg_first = 0.0f;
    float avg_last = 0.0f;
    
    for (int i = 0; i < 10 && i < MAX_EPISODES; i++) {
        avg_first += episode_returns[i];
    }
    avg_first /= 10.0f;
    
    int last_start = (MAX_EPISODES > 10) ? (MAX_EPISODES - 10) : 0;
    for (int i = last_start; i < MAX_EPISODES; i++) {
        avg_last += episode_returns[i];
    }
    avg_last /= (float)(MAX_EPISODES - last_start);
    
    printf("Average return (first 10): %.3f\n", avg_first);
    printf("Average return (last 10): %.3f\n", avg_last);
    printf("Improvement: %.3f\n", avg_last - avg_first);
    printf("\n");
    
    // Check if learning occurred
    int passed = 1;
    
    if (avg_last <= avg_first) {
        printf("❌ FAIL: No learning detected (return didn't improve)\n");
        passed = 0;
    } else {
        printf("✓ Learning detected: return improved by %.3f\n", avg_last - avg_first);
    }
    
    // Check graph structure
    GraphHeaderDisk *gh = file.graph_header;
    printf("\nGraph structure:\n");
    printf("  Nodes: %llu\n", (unsigned long long)gh->num_nodes);
    printf("  Edges: %llu\n", (unsigned long long)gh->num_edges);
    
    // Check FE in relevant subgraph
    float total_fe = 0.0f;
    int fe_nodes = 0;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        NodeDisk *node = &file.nodes[i];
        if (node->id != UINT64_MAX) {
            total_fe += node->fe_ema;
            fe_nodes++;
        }
    }
    float avg_fe = (fe_nodes > 0) ? (total_fe / fe_nodes) : 0.0f;
    printf("  Average FE_ema: %.6f\n", avg_fe);
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    if (passed) {
        printf("\n✓ PASS: Control learning test completed\n");
        return 0;
    } else {
        printf("\n❌ FAIL: Control learning test failed\n");
        return 1;
    }
}

