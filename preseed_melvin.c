/*
 * preseed_melvin.c - Preseed Melvin's brain with structural knowledge
 * 
 * Like evolution/development: Feed structured data that teaches Melvin
 * about his "body" - ports, patterns, basic structure.
 * 
 * Not hardcoding edges - teaching through experience.
 */

#include "src/melvin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* Preseed categories - what Melvin needs to know */
typedef struct {
    const char *name;
    const char **sequences;
    uint32_t count;
    float energy;
    uint32_t repetitions;  /* How many times to repeat (learning) */
} PreseedCategory;

/* PORT STRUCTURE - Teach Melvin about his input/output ports */
const char *port_structure[] = {
    "PORT_0_AUDIO_IN",
    "PORT_1_CAMERA_1", 
    "PORT_2_CAMERA_2",
    "PORT_100_AI_VISION",
    "PORT_101_AI_TEXT",
    "PORT_102_AI_AUDIO",
    "PORT_500_EXEC_OUT",
};

/* WORKING MEMORY - Teach concepts of storage/retrieval */
const char *working_memory[] = {
    "REMEMBER_THIS",
    "RECALL_THAT",
    "STORE_HERE",
    "FETCH_FROM",
    "WORKING_MEMORY_NODE",
};

/* ATTENTION - Teach focus/importance */
const char *attention_patterns[] = {
    "FOCUS_ON_THIS",
    "IMPORTANT_NOW",
    "ATTEND_HERE",
    "IGNORE_NOISE",
    "SALIENT_EVENT",
};

/* ERROR HANDLING - Teach failure patterns */
const char *error_patterns[] = {
    "ERROR_DETECTED",
    "FAILED_ATTEMPT", 
    "TRY_AGAIN",
    "SUCCESS_SIGNAL",
};

/* SEQUENCES - Common patterns Melvin will see */
const char *common_sequences[] = {
    "INPUT_PROCESS_OUTPUT",
    "SENSE_THINK_ACT",
    "OBSERVE_LEARN_REMEMBER",
    "QUESTION_SEARCH_ANSWER",
    "ACTIVATE_PROPAGATE_STABILIZE",
    "CAMERA_VISION_LABEL",
    "AUDIO_WHISPER_TEXT",
    "PATTERN_MATCH_ACTIVATE",
    "ENERGY_FLOWS_THROUGH_EDGES",
    "HIGH_ENERGY_EXPLORE_LEARN",
    "LOW_ENERGY_REST_CONSOLIDATE",
    "REPEAT_SEQUENCE_REMEMBER",
    "NEW_INPUT_CREATE_EDGE",
    "SIMILAR_NODES_CONNECT",
    "STRONG_PATTERN_PERSISTS",
};

/* META-STRUCTURE - Teach about graph itself */
const char *meta_structure[] = {
    "NODE_EDGE_PATTERN",
    "ACTIVATION_ENERGY_FLOW",
    "PATTERN_MATCHES_SEQUENCE",
    "EXEC_RUNS_CODE",
    "TOOL_CALL_RESULT",
};

void preseed_category(Graph *g, PreseedCategory *cat) {
    printf("\nPreseeding: %s (%u sequences, %u reps each)\n", 
           cat->name, cat->count, cat->repetitions);
    
    uint64_t start_edges = g->edge_count;
    
    /* Repeat to strengthen patterns */
    for (uint32_t rep = 0; rep < cat->repetitions; rep++) {
        for (uint32_t i = 0; i < cat->count; i++) {
            const char *seq = cat->sequences[i];
            
            /* Feed sequence byte by byte */
            for (size_t j = 0; j < strlen(seq); j++) {
                melvin_feed_byte(g, 0, (uint8_t)seq[j], cat->energy);
            }
            
            /* Feed delimiter to separate sequences */
            melvin_feed_byte(g, 0, '\n', cat->energy * 0.5f);
            
            /* Let UEL process */
            melvin_call_entry(g);
        }
    }
    
    uint64_t new_edges = g->edge_count - start_edges;
    printf("  Created %llu new edges\n", (unsigned long long)new_edges);
}

int main(int argc, char *argv[]) {
    const char *brain_path = (argc > 1) ? argv[1] : "brain_preseeded.m";
    
    printf("========================================\n");
    printf("MELVIN PRESEEDING SYSTEM\n");
    printf("========================================\n");
    printf("Teaching Melvin his structure through experience...\n\n");
    
    /* Create brain with organic sizing */
    uint32_t initial_nodes = 2000;      /* Enough for structure + growth */
    uint32_t edge_capacity = 50000;     /* Room for connections */
    uint32_t blob_size = 256 * 1024;    /* 256KB for patterns/code */
    
    printf("Creating brain: %u nodes, %u edge capacity, %u blob\n",
           initial_nodes, edge_capacity, blob_size);
    
    Graph *g = melvin_open(brain_path, initial_nodes, edge_capacity, blob_size);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    printf("Initial state: %llu nodes, %llu edges\n\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    
    /* Define preseeding categories with organic parameters */
    PreseedCategory categories[] = {
        {
            .name = "Port Structure",
            .sequences = port_structure,
            .count = sizeof(port_structure) / sizeof(port_structure[0]),
            .energy = 0.5f,  /* Strong - this is fundamental */
            .repetitions = 10  /* Repeat to strengthen */
        },
        {
            .name = "Working Memory",
            .sequences = working_memory,
            .count = sizeof(working_memory) / sizeof(working_memory[0]),
            .energy = 0.4f,
            .repetitions = 8
        },
        {
            .name = "Attention Patterns",
            .sequences = attention_patterns,
            .count = sizeof(attention_patterns) / sizeof(attention_patterns[0]),
            .energy = 0.4f,
            .repetitions = 7
        },
        {
            .name = "Error Handling",
            .sequences = error_patterns,
            .count = sizeof(error_patterns) / sizeof(error_patterns[0]),
            .energy = 0.5f,  /* Important - Melvin needs to handle errors */
            .repetitions = 12
        },
        {
            .name = "Common Sequences",
            .sequences = common_sequences,
            .count = sizeof(common_sequences) / sizeof(common_sequences[0]),
            .energy = 0.3f,
            .repetitions = 15  /* High repetition - will see these a lot */
        },
        {
            .name = "Meta Structure",
            .sequences = meta_structure,
            .count = sizeof(meta_structure) / sizeof(meta_structure[0]),
            .energy = 0.4f,
            .repetitions = 10
        }
    };
    
    uint32_t num_categories = sizeof(categories) / sizeof(categories[0]);
    
    /* Preseed each category */
    for (uint32_t i = 0; i < num_categories; i++) {
        preseed_category(g, &categories[i]);
    }
    
    /* Cross-connect categories by feeding combinations */
    printf("\nCreating cross-connections...\n");
    const char *combos[] = {
        "PORT_1_CAMERA_1 INPUT_PROCESS_OUTPUT PORT_100_AI_VISION",
        "PORT_0_AUDIO_IN INPUT_PROCESS_OUTPUT PORT_101_AI_TEXT",
        "FOCUS_ON_THIS IMPORTANT_NOW ATTEND_HERE",
        "ERROR_DETECTED TRY_AGAIN SUCCESS_SIGNAL",
        "REMEMBER_THIS WORKING_MEMORY_NODE RECALL_THAT",
        "OBSERVE_LEARN_REMEMBER STORE_HERE FETCH_FROM",
        "CAMERA_VISION_LABEL PORT_100_AI_VISION",
        "AUDIO_WHISPER_TEXT PORT_101_AI_TEXT",
        "HIGH_ENERGY_EXPLORE_LEARN NEW_INPUT_CREATE_EDGE",
        "PATTERN_MATCH_ACTIVATE ENERGY_FLOWS_THROUGH_EDGES",
        "REPEAT_SEQUENCE_REMEMBER STRONG_PATTERN_PERSISTS",
        "INPUT_PROCESS_OUTPUT SENSE_THINK_ACT",
        "NODE_EDGE_PATTERN ACTIVATE_PROPAGATE_STABILIZE",
        "SIMILAR_NODES_CONNECT STRONG_PATTERN_PERSISTS",
        "PORT_500_EXEC_OUT TOOL_CALL_RESULT",
    };
    
    for (size_t i = 0; i < sizeof(combos) / sizeof(combos[0]); i++) {
        for (size_t j = 0; j < strlen(combos[i]); j++) {
            melvin_feed_byte(g, 0, (uint8_t)combos[i][j], 0.4f);
        }
        melvin_feed_byte(g, 0, '\n', 0.2f);
        melvin_call_entry(g);
    }
    
    /* Final state */
    printf("\n========================================\n");
    printf("PRESEEDING COMPLETE\n");
    printf("========================================\n");
    printf("Final state:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("  Avg edge strength: %.4f\n", g->avg_edge_strength);
    printf("  Avg activation: %.4f\n", g->avg_activation);
    printf("\nBrain saved to: %s\n", brain_path);
    printf("\n✓ Melvin has been taught his structure!\n");
    printf("  - Knows his ports\n");
    printf("  - Understands working memory\n");
    printf("  - Has attention patterns\n");
    printf("  - Can handle errors\n");
    printf("  - Knows common sequences\n");
    printf("\nReady to learn from experience!\n");
    
    melvin_close(g);
    return 0;
}

