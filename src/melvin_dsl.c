// Graph DSL Interpreter
// Allows programming the graph as data/code instead of hardcoding in C
// This keeps C as pure hardware - all intelligence is expressed in DSL

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Simple DSL parser state
typedef struct {
    Graph *g;
    char *line;
    size_t line_cap;
} DSLState;

// Forward declarations
static int dsl_parse_pattern(DSLState *state, const char *line);
static int dsl_parse_output_pattern(DSLState *state, const char *line);
static int dsl_parse_feed(DSLState *state, const char *line);
static int dsl_parse_learn(DSLState *state, const char *line);
static int dsl_parse_show(DSLState *state, const char *line);
static int dsl_parse_output(DSLState *state, const char *line);
static void dsl_help(void);

// Skip whitespace
static const char *skip_ws(const char *s) {
    while (*s && isspace(*s)) s++;
    return s;
}

// Parse a quoted string
static int parse_quoted_string(const char *s, char *out, size_t out_len) {
    s = skip_ws(s);
    if (*s != '"') return 0;
    s++;
    
    size_t i = 0;
    while (*s && *s != '"' && i < out_len - 1) {
        if (*s == '\\' && s[1]) {
            s++;
            if (*s == 'n') out[i++] = '\n';
            else if (*s == 't') out[i++] = '\t';
            else out[i++] = *s;
        } else {
            out[i++] = *s;
        }
        s++;
    }
    out[i] = '\0';
    return (*s == '"') ? 1 : 0;
}

// Parse pattern definition: pattern "name" { atom 0: byte 'x'; ... }
static int dsl_parse_pattern(DSLState *state, const char *line) {
    // Skip "pattern"
    line = skip_ws(line + 7);
    
    // Parse pattern name (quoted string)
    char pattern_name[256];
    if (!parse_quoted_string(line, pattern_name, sizeof(pattern_name))) {
        fprintf(stderr, "Error: Expected quoted pattern name\n");
        return 0;
    }
    
    line = strchr(line, '"');
    if (!line) return 0;
    line = strchr(line + 1, '{');
    if (!line) {
        fprintf(stderr, "Error: Expected '{' after pattern name\n");
        return 0;
    }
    line++;
    
    // Simple parser for atoms (for now, just support byte atoms)
    PatternAtom atoms[16];
    size_t num_atoms = 0;
    
    // Parse atoms (simplified - just look for "atom N: byte 'X'")
    const char *atom_start = line;
    while (num_atoms < 16 && (atom_start = strstr(atom_start, "atom"))) {
        atom_start += 4;
        atom_start = skip_ws(atom_start);
        
        // Parse atom index
        int delta = 0;
        if (sscanf(atom_start, "%d", &delta) != 1) break;
        
        // Find ':'
        atom_start = strchr(atom_start, ':');
        if (!atom_start) break;
        atom_start++;
        atom_start = skip_ws(atom_start);
        
        // Check for "byte"
        if (strncmp(atom_start, "byte", 4) != 0) break;
        atom_start += 4;
        atom_start = skip_ws(atom_start);
        
        // Parse byte value: 'X'
        if (*atom_start != '\'') break;
        atom_start++;
        uint8_t byte_val = (uint8_t)*atom_start;
        atom_start += 2;  // skip ' and potential '
        
        atoms[num_atoms].delta = (int16_t)delta;
        atoms[num_atoms].mode = 0;  // CONST_BYTE
        atoms[num_atoms].value = byte_val;
        num_atoms++;
    }
    
    if (num_atoms == 0) {
        fprintf(stderr, "Error: No atoms found in pattern\n");
        return 0;
    }
    
    // Create pattern in graph
    Node *pattern = graph_add_pattern(state->g, atoms, num_atoms, 0.5f);
    if (pattern) {
        printf("Created pattern \"%s\" (id: %llu, %zu atoms)\n",
               pattern_name, (unsigned long long)pattern->id, num_atoms);
        return 1;
    }
    
    return 0;
}

// Parse output_pattern command: output_pattern "name" when pattern_id=X output byte 'Y'
// Creates a pattern that connects internal pattern activation to output nodes
static int dsl_parse_output_pattern(DSLState *state, const char *line) {
    line = skip_ws(line + 14);  // Skip "output_pattern"
    
    // Parse pattern name
    char pattern_name[256];
    if (!parse_quoted_string(line, pattern_name, sizeof(pattern_name))) {
        fprintf(stderr, "Error: Expected quoted pattern name\n");
        return 0;
    }
    
    // Find "when" after the quoted name
    const char *when_pos = strstr(line, "when");
    if (!when_pos) {
        fprintf(stderr, "Error: Expected 'when pattern_id=X'\n");
        return 0;
    }
    line = when_pos + 4;
    line = skip_ws(line);
    
    // Parse "pattern_id=X"
    uint64_t trigger_pattern_id = 0;
    if (sscanf(line, "pattern_id=%llu", (unsigned long long *)&trigger_pattern_id) != 1) {
        fprintf(stderr, "Error: Expected 'pattern_id=N'\n");
        return 0;
    }
    
    line = strstr(line, "output");
    if (!line) {
        fprintf(stderr, "Error: Expected 'output byte ...'\n");
        return 0;
    }
    line += 6;
    line = skip_ws(line);
    
    // Parse "byte 'X'"
    if (strncmp(line, "byte", 4) != 0) {
        fprintf(stderr, "Error: Expected 'byte'\n");
        return 0;
    }
    line += 4;
    line = skip_ws(line);
    
    if (*line != '\'') {
        fprintf(stderr, "Error: Expected quoted byte value (got '%c')\n", *line);
        return 0;
    }
    uint8_t output_byte = (uint8_t)line[1];
    
    // Find or create the trigger pattern
    Node *trigger_pattern = graph_find_node_by_id(state->g, trigger_pattern_id);
    if (!trigger_pattern || trigger_pattern->kind != NODE_PATTERN) {
        fprintf(stderr, "Error: Pattern id %llu not found\n", 
                (unsigned long long)trigger_pattern_id);
        return 0;
    }
    
    // Find or create output node
    Node *output_node = NULL;
    // Search for existing output node with this byte
    for (uint64_t i = 0; i < state->g->num_nodes; i++) {
        Node *n = &state->g->nodes[i];
        if (n->kind == NODE_OUTPUT && n->payload_len > 0) {
            uint8_t b = state->g->blob[n->payload_offset];
            if (b == output_byte) {
                output_node = n;
                break;
            }
        }
    }
    
    if (!output_node) {
        // Graph-native: create OUTPUT node using hardware function
        static uint64_t next_output_id = (1ULL << 62);
        uint8_t payload = output_byte;
        output_node = graph_create_node(state->g, NODE_OUTPUT, next_output_id++, &payload, 1);
        if (!output_node) {
            fprintf(stderr, "Error: Failed to create output node\n");
            return 0;
        }
    }
    
    // Create edge from trigger pattern to output node
    // This is the output pattern: when pattern is active, activate output
    Edge *e = graph_add_edge(state->g, trigger_pattern_id, output_node->id, 1.0f);
    if (e) {
        printf("Created output pattern \"%s\": pattern %llu -> output '%c' (node %llu)\n",
               pattern_name, (unsigned long long)trigger_pattern_id, 
               (char)output_byte, (unsigned long long)output_node->id);
        return 1;
    }
    
    return 0;
}

// Parse feed command: feed "data string"
static int dsl_parse_feed(DSLState *state, const char *line) {
    line = skip_ws(line + 4);
    
    char data[1024];
    if (!parse_quoted_string(line, data, sizeof(data))) {
        fprintf(stderr, "Error: Expected quoted data string\n");
        return 0;
    }
    
    // Feed as DATA nodes
    uint64_t prev_id = UINT64_MAX;
    size_t count = 0;
    for (size_t i = 0; data[i]; i++) {
        Node *n = graph_add_data_byte(state->g, (uint8_t)data[i]);
        if (n) {
            if (prev_id != UINT64_MAX) {
                graph_add_edge(state->g, prev_id, n->id, 1.0f);
            }
            prev_id = n->id;
            count++;
        }
    }
    
    printf("Fed %zu bytes as DATA nodes\n", count);
    return 1;
}

// Parse learn command: learn [episodes=N] [threshold=X]
static int dsl_parse_learn(DSLState *state, const char *line) {
    line = skip_ws(line + 5);
    
    int episodes = 10;
    float threshold = 0.8f;
    float lr_q = 0.2f;
    
    // Simple parameter parsing
    if (strstr(line, "episodes=")) {
        sscanf(strstr(line, "episodes="), "episodes=%d", &episodes);
    }
    if (strstr(line, "threshold=")) {
        sscanf(strstr(line, "threshold="), "threshold=%f", &threshold);
    }
    
    // Find all PATTERN nodes
    Node *patterns[64];
    size_t num_patterns = 0;
    
    for (uint64_t i = 0; i < state->g->num_nodes && num_patterns < 64; i++) {
        if (state->g->nodes[i].kind == NODE_PATTERN) {
            patterns[num_patterns++] = &state->g->nodes[i];
        }
    }
    
    if (num_patterns == 0) {
        fprintf(stderr, "Error: No patterns found. Create patterns first.\n");
        return 0;
    }
    
    uint64_t start_id = 0;
    uint64_t end_id = state->g->next_data_pos > 0 ? state->g->next_data_pos - 1 : 0;
    
    printf("Running %d learning episodes with %zu patterns...\n", episodes, num_patterns);
    
    for (int it = 0; it < episodes; it++) {
        float err = graph_self_consistency_episode_multi_pattern(
            state->g, patterns, num_patterns, start_id, end_id, threshold, lr_q);
        printf("  Episode %d: error=%.4f\n", it, err);
    }
    
    return 1;
}

// Parse show command: show [patterns|bindings|stats]
static int dsl_parse_show(DSLState *state, const char *line) {
    line = skip_ws(line + 4);
    
    if (strncmp(line, "patterns", 8) == 0) {
        printf("=== Patterns ===\n");
        for (uint64_t i = 0; i < state->g->num_nodes; i++) {
            Node *p = &state->g->nodes[i];
            if (p->kind == NODE_PATTERN) {
                printf("Pattern id=%llu, q=%.4f\n",
                       (unsigned long long)p->id, p->q);
            }
        }
        return 1;
    }
    
    if (strncmp(line, "stats", 5) == 0) {
        graph_print_stats(state->g);
        return 1;
    }
    
    if (strncmp(line, "bindings", 8) == 0) {
        graph_debug_print_pattern_bindings(state->g, 10);
        return 1;
    }
    
    if (strncmp(line, "outputs", 7) == 0) {
        printf("=== Output Nodes ===\n");
        int count = 0;
        for (uint64_t i = 0; i < state->g->num_nodes; i++) {
            Node *n = &state->g->nodes[i];
            if (n->kind == NODE_OUTPUT) {
                if (n->payload_len > 0) {
                    uint8_t b = state->g->blob[n->payload_offset];
                    printf("OUTPUT id=%llu byte='%c' (0x%02X) activation=%.3f\n",
                           (unsigned long long)n->id, 
                           (b >= 32 && b <= 126) ? (char)b : '.',
                           b, n->a);
                    count++;
                }
            }
        }
        if (count == 0) {
            printf("(no output nodes created yet)\n");
        }
        return 1;
    }
    
    fprintf(stderr, "Error: Unknown show command. Use: patterns, stats, bindings, or outputs\n");
    return 0;
}

// Activate patterns that match current input
static void dsl_activate_matching_patterns(DSLState *state) {
    if (!state->g) return;
    
    // Find all PATTERN nodes
    Node *patterns[64];
    size_t num_patterns = 0;
    
    for (uint64_t i = 0; i < state->g->num_nodes && num_patterns < 64; i++) {
        if (state->g->nodes[i].kind == NODE_PATTERN) {
            patterns[num_patterns++] = &state->g->nodes[i];
        }
    }
    
    // Activate patterns based on their match scores
    // Use recent data nodes (last N nodes fed)
    uint64_t start_id = state->g->next_data_pos > 10 ? 
                        state->g->next_data_pos - 10 : 0;
    uint64_t end_id = state->g->next_data_pos > 0 ? 
                      state->g->next_data_pos - 1 : 0;
    
    fprintf(stderr, "DEBUG: Checking patterns against data nodes %llu-%llu (next_data_pos=%llu)\n",
            (unsigned long long)start_id, (unsigned long long)end_id,
            (unsigned long long)state->g->next_data_pos);
    
    for (size_t p = 0; p < num_patterns; p++) {
        Node *pattern = patterns[p];
        float max_score = 0.0f;
        uint64_t best_anchor = UINT64_MAX;
        
        // Find best match score across recent anchor positions
        for (uint64_t anchor = start_id; anchor <= end_id; anchor++) {
            float score = pattern_match_score(state->g, pattern, anchor);
            if (score > max_score) {
                max_score = score;
                best_anchor = anchor;
            }
        }
        
        // Activate pattern based on match score and quality
        // High quality + good match = high activation
        // Use match score directly (it's already normalized)
        pattern->a = max_score * pattern->q;
        
        // Debug: show activation
        fprintf(stderr, "Pattern %llu: match_score=%.3f, q=%.3f, activation=%.3f (best_anchor=%llu)\n",
                (unsigned long long)pattern->id, max_score, pattern->q, 
                pattern->a, (unsigned long long)best_anchor);
    }
}

// Parse output command: output [threshold=X]
// Generates output by propagating activation from patterns to output nodes
static int dsl_parse_output(DSLState *state, const char *line) {
    line = skip_ws(line + 6);  // Skip "output"
    
    float threshold = 0.3f;  // Lower threshold for outputs
    if (strstr(line, "threshold=")) {
        sscanf(strstr(line, "threshold="), "threshold=%f", &threshold);
    }
    
    // Step 1: Activate patterns that match current input
    dsl_activate_matching_patterns(state);
    
    // Step 2: Propagate activation from patterns to outputs
    graph_propagate(state->g, 3);
    
    // Step 3: Collect active output nodes (sorted by activation)
    typedef struct {
        Node *node;
        float activation;
    } OutputCandidate;
    
    OutputCandidate candidates[256];
    size_t num_candidates = 0;
    
    for (uint64_t i = 0; i < state->g->num_nodes && num_candidates < 256; i++) {
        Node *n = &state->g->nodes[i];
        if (n->kind == NODE_OUTPUT && n->a > threshold) {
            if (n->payload_len > 0) {
                candidates[num_candidates].node = n;
                candidates[num_candidates].activation = n->a;
                num_candidates++;
            }
        }
    }
    
    // Sort by activation (descending)
    for (size_t i = 0; i < num_candidates; i++) {
        for (size_t j = i + 1; j < num_candidates; j++) {
            if (candidates[j].activation > candidates[i].activation) {
                OutputCandidate tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }
    
    // Step 4: Emit outputs
    printf("Output: ");
    int has_output = 0;
    for (size_t i = 0; i < num_candidates; i++) {
        uint8_t b = state->g->blob[candidates[i].node->payload_offset];
        putchar((char)b);
        has_output = 1;
    }
    
    if (!has_output) {
        printf("(no output - patterns may not be active or threshold too high)");
    }
    printf("\n");
    
    return 1;
}

static void dsl_help(void) {
    printf("Graph DSL Commands:\n");
    printf("  pattern \"name\" { atom 0: byte 'x'; atom 1: byte 'y'; }\n");
    printf("  output_pattern \"name\" when pattern_id=X output byte 'Y'\n");
    printf("  feed \"data string\"\n");
    printf("  learn [episodes=N] [threshold=X]\n");
    printf("  output [threshold=X]  # Generate output from graph state\n");
    printf("  show [patterns|stats|bindings|outputs]\n");
    printf("  help\n");
    printf("  quit\n");
}

// Main DSL REPL
int main(int argc, char *argv[]) {
    // Create graph
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    if (!g) {
        fprintf(stderr, "Failed to create graph\n");
        return 1;
    }
    
    DSLState state = {.g = g, .line = NULL, .line_cap = 0};
    
    printf("Melvin Graph DSL Terminal\n");
    printf("Type 'help' for commands, 'quit' to exit\n");
    printf("> ");
    
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), stdin)) {
        // Remove newline
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
            len--;
        }
        
        if (len == 0) {
            printf("> ");
            continue;
        }
        
        const char *cmd = skip_ws(buffer);
        
        // Parse commands
        if (strncmp(cmd, "pattern", 7) == 0) {
            dsl_parse_pattern(&state, cmd);
        } else if (strncmp(cmd, "output_pattern", 14) == 0) {
            dsl_parse_output_pattern(&state, cmd);
        } else if (strncmp(cmd, "feed", 4) == 0) {
            dsl_parse_feed(&state, cmd);
        } else if (strncmp(cmd, "learn", 5) == 0) {
            dsl_parse_learn(&state, cmd);
        } else if (strncmp(cmd, "output", 6) == 0) {
            dsl_parse_output(&state, cmd);
        } else if (strncmp(cmd, "show", 4) == 0) {
            dsl_parse_show(&state, cmd);
        } else if (strncmp(cmd, "help", 4) == 0) {
            dsl_help();
        } else if (strncmp(cmd, "quit", 4) == 0 || strncmp(cmd, "exit", 4) == 0) {
            break;
        } else {
            fprintf(stderr, "Unknown command. Type 'help' for commands.\n");
        }
        
        printf("> ");
    }
    
    graph_destroy(g);
    return 0;
}

