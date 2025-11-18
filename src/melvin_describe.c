#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Produce a readable "self-report" on an input string
// Shows what patterns explain the input and how

void print_pattern_atoms_readable(const Graph *g, const Node *pattern) {
    if (!g || !pattern || pattern->kind != NODE_PATTERN) return;
    
    size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
    if (num_atoms == 0) return;
    
    const PatternAtom *atoms =
        (const PatternAtom *)(g->blob + pattern->payload_offset);
    
    printf("    Pattern: ");
    for (size_t i = 0; i < num_atoms; i++) {
        if (atoms[i].mode == 0) { // CONST_BYTE
            char c = (atoms[i].value >= 32 && atoms[i].value <= 126) 
                    ? (char)atoms[i].value : '.';
            if (i > 0) printf(" ");
            printf("[%+d]='%c'", atoms[i].delta, c);
        } else { // BLANK
            if (i > 0) printf(" ");
            printf("[%+d]=_", atoms[i].delta);
        }
    }
    printf("\n");
}

void describe_input(const char *input_str) {
    if (!input_str || strlen(input_str) == 0) {
        printf("Empty input\n");
        return;
    }
    
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    if (!g) {
        fprintf(stderr, "Failed to create graph\n");
        return;
    }
    
    // Feed input as DATA nodes
    size_t len = strlen(input_str);
    uint64_t prev_data_id = UINT64_MAX;
    uint64_t start_id = UINT64_MAX;
    
    for (size_t i = 0; i < len; i++) {
        Node *data_node = graph_add_data_byte(g, (uint8_t)input_str[i]);
        if (data_node) {
            if (start_id == UINT64_MAX) {
                start_id = data_node->id;
            }
            if (prev_data_id != UINT64_MAX) {
                graph_add_edge(g, prev_data_id, data_node->id, 1.0f);
            }
            prev_data_id = data_node->id;
        }
    }
    
    uint64_t end_id = prev_data_id;
    
    // Create patterns (bigrams and trigrams)
    Node *patterns[32];
    size_t num_patterns = 0;
    
    for (size_t i = 0; i < len - 1 && num_patterns < 32; i++) {
        PatternAtom atoms[2];
        atoms[0].delta = 0;
        atoms[0].mode = 0; // CONST_BYTE
        atoms[0].value = (uint8_t)input_str[i];
        atoms[1].delta = 1;
        atoms[1].mode = 0;
        atoms[1].value = (uint8_t)input_str[i + 1];
        
        Node *pattern = graph_add_pattern(g, atoms, 2, 0.5f);
        if (pattern) {
            patterns[num_patterns++] = pattern;
        }
    }
    
    for (size_t i = 0; i < len - 2 && num_patterns < 32; i++) {
        PatternAtom atoms[3];
        atoms[0].delta = 0;
        atoms[0].mode = 0;
        atoms[0].value = (uint8_t)input_str[i];
        atoms[1].delta = 1;
        atoms[1].mode = 0;
        atoms[1].value = (uint8_t)input_str[i + 1];
        atoms[2].delta = 2;
        atoms[2].mode = 0;
        atoms[2].value = (uint8_t)input_str[i + 2];
        
        Node *pattern = graph_add_pattern(g, atoms, 3, 0.5f);
        if (pattern) {
            patterns[num_patterns++] = pattern;
        }
    }
    
    // Run learning episodes
    for (int iter = 0; iter < 3; iter++) {
        graph_self_consistency_episode_multi_pattern(g,
                                                     patterns,
                                                     num_patterns,
                                                     start_id,
                                                     end_id,
                                                     0.8f,  // match_threshold
                                                     0.2f); // lr_q
    }
    
    // Collect patterns that have bindings
    typedef struct {
        uint64_t pattern_id;
        double q;
        int binding_count;
        Node *pattern_node;
    } PatternInfo;
    
    PatternInfo *pattern_infos = NULL;
    size_t num_pattern_infos = 0;
    size_t cap_pattern_infos = 0;
    
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].kind != NODE_PATTERN) continue;
        
        Node *p = &g->nodes[i];
        
        // Count bindings
        int bindings = 0;
        for (uint64_t e = 0; e < g->num_edges; e++) {
            if (g->edges[e].src == p->id) {
                Node *d = graph_find_node_by_id(g, g->edges[e].dst);
                if (d && d->kind == NODE_DATA) {
                    bindings++;
                }
            }
        }
        
        if (bindings > 0) {
            if (num_pattern_infos >= cap_pattern_infos) {
                cap_pattern_infos = cap_pattern_infos ? cap_pattern_infos * 2 : 16;
                pattern_infos = realloc(pattern_infos, 
                                       cap_pattern_infos * sizeof(PatternInfo));
            }
            
            pattern_infos[num_pattern_infos].pattern_id = p->id;
            pattern_infos[num_pattern_infos].q = p->q;
            pattern_infos[num_pattern_infos].binding_count = bindings;
            pattern_infos[num_pattern_infos].pattern_node = p;
            num_pattern_infos++;
        }
    }
    
    // Sort by binding count (descending)
    for (size_t i = 0; i < num_pattern_infos; i++) {
        for (size_t j = i + 1; j < num_pattern_infos; j++) {
            if (pattern_infos[j].binding_count > pattern_infos[i].binding_count) {
                PatternInfo tmp = pattern_infos[i];
                pattern_infos[i] = pattern_infos[j];
                pattern_infos[j] = tmp;
            }
        }
    }
    
    // Print report
    printf("Input: '%s'\n", input_str);
    printf("Length: %zu bytes\n", len);
    printf("\nTop patterns explaining this input:\n\n");
    
    int shown = 0;
    for (size_t i = 0; i < num_pattern_infos && shown < 10; i++) {
        PatternInfo *info = &pattern_infos[i];
        printf("  Pattern %llu:\n", (unsigned long long)info->pattern_id);
        printf("    Quality: %.3f\n", info->q);
        printf("    Bindings: %d\n", info->binding_count);
        print_pattern_atoms_readable(g, info->pattern_node);
        
        // Show which positions it binds to
        printf("    Applied at positions: ");
        int pos_shown = 0;
        for (uint64_t e = 0; e < g->num_edges && pos_shown < 10; e++) {
            if (g->edges[e].src == info->pattern_id) {
                Node *d = graph_find_node_by_id(g, g->edges[e].dst);
                if (d && d->kind == NODE_DATA) {
                    if (pos_shown > 0) printf(", ");
                    printf("%llu", (unsigned long long)d->id);
                    pos_shown++;
                }
            }
        }
        if (pos_shown >= 10) printf(" ...");
        printf("\n\n");
        shown++;
    }
    
    if (num_pattern_infos == 0) {
        printf("  (No patterns with bindings found)\n");
    }
    
    free(pattern_infos);
    graph_destroy(g);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        // Read from stdin
        char *line = NULL;
        size_t len = 0;
        ssize_t read;
        
        printf("Reading from stdin (Ctrl+D to finish):\n\n");
        while ((read = getline(&line, &len, stdin)) != -1) {
            if (read > 0 && line[read - 1] == '\n') {
                line[read - 1] = '\0';
            }
            if (strlen(line) > 0) {
                describe_input(line);
                printf("\n");
            }
        }
        free(line);
    } else {
        // Process each argument as input
        for (int i = 1; i < argc; i++) {
            describe_input(argv[i]);
            if (i < argc - 1) {
                printf("\n");
            }
        }
    }
    
    return 0;
}

