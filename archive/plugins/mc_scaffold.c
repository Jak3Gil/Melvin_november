#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include "mc_scaffold.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

// External helpers (from melvin.c)
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

// MC table access (from melvin.c)
typedef void (*MCFn)(Brain *, uint64_t);
typedef struct {
    const char *name;
    MCFn fn;
    uint32_t flags;
} MCEntry;
extern MCEntry g_mc_table[];
extern uint32_t g_mc_count;

// Static state
static int scaffolds_applied = 0;

// Helper: Extract token from string (for parsing context/effect fields)
static const char *skip_whitespace(const char *s) {
    while (*s && isspace(*s)) s++;
    return s;
}

static const char *parse_identifier(const char *s, char *out, size_t out_size) {
    s = skip_whitespace(s);
    size_t i = 0;
    while (*s && (isalnum(*s) || *s == '_' || *s == ':' || *s == '>' || *s == '<') && i < out_size - 1) {
        out[i++] = *s++;
    }
    out[i] = '\0';
    return s;
}

// Helper: Create or find a channel node
static uint64_t ensure_channel_node(Brain *g, uint32_t channel_id) {
    // For simplicity, use value field to store channel ID
    // In a real system, we'd have a more robust lookup
    uint64_t n = g->header->num_nodes;
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &g->nodes[i];
        if (node->kind == NODE_KIND_TAG && (uint32_t)node->value == channel_id) {
            return i;
        }
    }
    
    // Create new channel node
    uint64_t ch_node = alloc_node(g);
    if (ch_node == UINT64_MAX) return UINT64_MAX;
    
    Node *ch = &g->nodes[ch_node];
    ch->kind = NODE_KIND_TAG;
    ch->a = 0.5f;
    ch->value = (float)channel_id;
    ch->flags = 0;
    
    return ch_node;
}

// Helper: Create or find a blank node for variable
static uint64_t ensure_blank_node(Brain *g, const char *var_name) {
    // Hash the variable name to check if it exists
    uint32_t hash = 0;
    size_t len = strlen(var_name);
    for (size_t i = 0; i < len && i < 32; i++) {
        hash = hash * 31 + (unsigned char)var_name[i];
    }
    
    // Use find_or_create_node to prevent duplicates
    // Nodes should be reused, not duplicated - edges connect them to contexts
    extern uint64_t find_or_create_node(Brain *, uint32_t hash, uint32_t kind);
    uint64_t blank_node = find_or_create_node(g, hash, NODE_KIND_BLANK);
    
    if (blank_node != UINT64_MAX && blank_node != 0) {
        Node *blank = &g->nodes[blank_node];
        // If node was just created, set default activation
        if (blank->a == 0.0f) {
            blank->a = 0.3f;
            blank->flags = 0;
        }
    }
    
    return blank_node;
}

// Parse context field: "vision:HUMAN_LIMB, sensor:TORQUE>THRESH, motor:JOINT_ID"
static void parse_context_fields(Brain *g, uint64_t pattern_node, const char *context_str) {
    const char *p = context_str;
    
    while (*p) {
        p = skip_whitespace(p);
        if (!*p || *p == '}') break;
        
        // Parse channel:variable
        char channel_name[64];
        p = parse_identifier(p, channel_name, sizeof(channel_name));
        
        if (*p == ':') {
            p++; // skip ':'
            
            // Determine channel ID
            uint32_t channel_id = CH_META;
            if (strncmp(channel_name, "vision", 6) == 0) {
                channel_id = CH_VISION;
            } else if (strncmp(channel_name, "sensor", 6) == 0) {
                channel_id = CH_SENSOR;
            } else if (strncmp(channel_name, "motor", 5) == 0) {
                channel_id = CH_MOTOR;
            }
            
            // Get channel node
            uint64_t ch_node = ensure_channel_node(g, channel_id);
            if (ch_node == UINT64_MAX) break;
            
            // Parse variable name (may include operators like >THRESH)
            char var_name[128];
            p = parse_identifier(p, var_name, sizeof(var_name));
            
            // Create blank node for variable
            uint64_t blank_node = ensure_blank_node(g, var_name);
            if (blank_node == UINT64_MAX) break;
            
            // Connect: CHANNEL -> BLANK -> PATTERN
            add_edge(g, ch_node, blank_node, 1.0f, EDGE_FLAG_CHAN | EDGE_FLAG_BIND);
            add_edge(g, blank_node, pattern_node, 1.0f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
            
            // Link to pattern with semantic edge
            add_edge(g, pattern_node, blank_node, 0.8f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
        }
        
        // Skip comma or closing brace
        if (*p == ',') p++;
        else if (*p == '}') break;
    }
}

// Parse effect field: "block_motor:JOINT_ID, reward:-10"
static void parse_effect_fields(Brain *g, uint64_t pattern_node, const char *effect_str) {
    const char *p = effect_str;
    
    while (*p) {
        p = skip_whitespace(p);
        if (!*p || *p == '}') break;
        
        // Parse action:target or reward:value
        char action_name[128];
        p = parse_identifier(p, action_name, sizeof(action_name));
        
        if (*p == ':') {
            p++; // skip ':'
            
            // Check if it's a reward
            if (strncmp(action_name, "reward", 6) == 0) {
                // Parse numeric value (could be negative)
                float reward_val = 0.0f;
                int sign = 1;
                if (*p == '-') {
                    sign = -1;
                    p++;
                }
                while (isdigit(*p)) {
                    reward_val = reward_val * 10.0f + (*p - '0');
                    p++;
                }
                reward_val *= sign;
                
                // Get reward channel
                uint64_t reward_ch = ensure_channel_node(g, CH_REWARD);
                if (reward_ch != UINT64_MAX) {
                    // Connect pattern to reward channel with weight
                    add_edge(g, pattern_node, reward_ch, reward_val, EDGE_FLAG_CHAN | EDGE_FLAG_CONTROL);
                }
            } else {
                // Other actions like block_motor:JOINT_ID
                // Parse target variable
                char target_var[128];
                p = parse_identifier(p, target_var, sizeof(target_var));
                
                // Create blank for target
                uint64_t target_blank = ensure_blank_node(g, target_var);
                if (target_blank == UINT64_MAX) break;
                
                // Determine output channel based on action
                uint32_t out_channel = CH_MOTOR;
                if (strncmp(action_name, "block_motor", 11) == 0) {
                    out_channel = CH_MOTOR;
                } else if (strncmp(action_name, "adjust_path", 11) == 0) {
                    out_channel = CH_VISION; // Could be different
                }
                
                // Get output channel node
                uint64_t out_ch = ensure_channel_node(g, out_channel);
                if (out_ch != UINT64_MAX) {
                    // Connect: PATTERN -> BLANK -> OUTPUT_CHANNEL (gating)
                    add_edge(g, pattern_node, target_blank, 1.0f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
                    add_edge(g, target_blank, out_ch, 0.9f, EDGE_FLAG_CONTROL | EDGE_FLAG_CHAN);
                }
            }
        }
        
        // Skip comma or closing brace
        if (*p == ',') p++;
        else if (*p == '}') break;
    }
}

// Main function: Emit a rule as a pattern into the graph
void mc_scaffold_emit_rule(Brain *g, const ScannedRule *r) {
    printf("[mc_scaffold] Emitting rule: %s\n", r->name);
    
    // (A) Create PATTERN_ROOT node for this rule
    uint64_t pattern_node = alloc_node(g);
    if (pattern_node == UINT64_MAX) {
        fprintf(stderr, "[mc_scaffold] Failed to allocate pattern node\n");
        return;
    }
    
    Node *pattern = &g->nodes[pattern_node];
    pattern->kind = NODE_KIND_PATTERN_ROOT;
    pattern->a = 0.6f;
    pattern->bias = 0.5f;
    pattern->reliability = 0.8f;
    
    // Store rule name hash in value
    uint32_t name_hash = 0;
    size_t name_len = strlen(r->name);
    for (size_t i = 0; i < name_len && i < 32; i++) {
        name_hash = name_hash * 31 + (unsigned char)r->name[i];
    }
    pattern->value = (float)name_hash;
    pattern->flags = ORIGIN_SCAFFOLD; // Mark as scaffold-originated
    
    // Store origin file in a connected META node
    uint64_t origin_node = alloc_node(g);
    if (origin_node != UINT64_MAX) {
        Node *orig = &g->nodes[origin_node];
        orig->kind = NODE_KIND_META;
        orig->a = 0.4f;
        
        // Hash filename
        uint32_t file_hash = 0;
        size_t file_len = strlen(r->origin_file);
        for (size_t i = 0; i < file_len && i < 32; i++) {
            file_hash = file_hash * 31 + (unsigned char)r->origin_file[i];
        }
        orig->value = (float)file_hash;
        
        add_edge(g, pattern_node, origin_node, 1.0f, EDGE_FLAG_REL);
    }
    
    printf("[mc_scaffold] Pattern node %llu created for rule %s (kind=%u, value=%.0f)\n", 
           (unsigned long long)pattern_node, r->name, pattern->kind, pattern->value);
    
    // (B) Create BLANK nodes for variables (done during context/effect parsing)
    
    // (C) Parse context and create input-side edges
    uint64_t context_edges_before = g->header->num_edges;
    if (r->context[0]) {
        printf("[mc_scaffold] Parsing context: %s\n", r->context);
        parse_context_fields(g, pattern_node, r->context);
        uint64_t context_edges_after = g->header->num_edges;
        printf("[mc_scaffold] Created %llu context edges for rule %s\n",
               (unsigned long long)(context_edges_after - context_edges_before), r->name);
    }
    
    // (D) Parse effect and create effect-side edges
    uint64_t effect_edges_before = g->header->num_edges;
    if (r->effect[0]) {
        printf("[mc_scaffold] Parsing effect: %s\n", r->effect);
        parse_effect_fields(g, pattern_node, r->effect);
        uint64_t effect_edges_after = g->header->num_edges;
        printf("[mc_scaffold] Created %llu effect edges for rule %s\n",
               (unsigned long long)(effect_edges_after - effect_edges_before), r->name);
    }
    
    // Verify pattern structure was created
    uint64_t pattern_edges_count = 0;
    for (uint64_t i = 0; i < g->header->num_edges; i++) {
        if (g->edges[i].src == pattern_node && (g->edges[i].flags & EDGE_FLAG_PATTERN)) {
            pattern_edges_count++;
        }
    }
    printf("[mc_scaffold] Rule %s fully emitted: pattern_node=%llu, pattern_edges=%llu\n", 
           r->name, (unsigned long long)pattern_node, (unsigned long long)pattern_edges_count);
}

// Check if scaffolds should be scanned (guard flag)
int mc_scaffold_should_scan(Brain *g) {
    // Check for scaffolds_applied flag in graph metadata
    // Look for a META node with specific value
    uint64_t n = g->header->num_nodes;
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &g->nodes[i];
        if (node->kind == NODE_KIND_META && (uint32_t)node->value == 0x53434146) { // "SCAF"
            scaffolds_applied = 1;
            return 0; // Skip scanning
        }
    }
    return 1; // Should scan
}

// Mark scaffolds as applied
static void mark_scaffolds_applied(Brain *g) {
    // Create a META node marking scaffolds as applied
    uint64_t flag_node = alloc_node(g);
    if (flag_node != UINT64_MAX) {
        Node *flag = &g->nodes[flag_node];
        flag->kind = NODE_KIND_META;
        flag->a = 1.0f;
        flag->value = 0x53434146; // "SCAF" - scaffolds applied flag
        scaffolds_applied = 1;
        printf("[mc_scaffold] Marked scaffolds as applied (flag node %llu)\n", 
               (unsigned long long)flag_node);
    }
}

// Cleanup: Delete all scaffold files
void mc_scaffold_cleanup(void) {
    printf("[mc_scaffold] Cleaning up scaffold files...\n");
    
    DIR *dir = opendir("scaffolds");
    if (!dir) {
        printf("[mc_scaffold] No scaffolds directory found\n");
        return;
    }
    
    struct dirent *ent;
    int deleted = 0;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        
        // Only delete .c files
        size_t len = strlen(ent->d_name);
        if (len > 2 && strcmp(ent->d_name + len - 2, ".c") == 0) {
            char path[512];
            snprintf(path, sizeof(path), "scaffolds/%s", ent->d_name);
            
            if (unlink(path) == 0) {
                printf("[mc_scaffold] Deleted: %s\n", path);
                deleted++;
            } else {
                perror("unlink");
            }
        }
    }
    closedir(dir);
    
    printf("[mc_scaffold] Cleanup complete: %d files deleted\n", deleted);
}

// Process a single scaffold file
void mc_scaffold_process_file(Brain *g, const char *file_path) {
    printf("[mc_scaffold] Processing scaffold file: %s\n", file_path);
    
    FILE *f = fopen(file_path, "r");
    if (!f) {
        perror("fopen scaffold");
        return;
    }
    
    // Read entire file
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size <= 0 || size > 1024 * 1024) { // Max 1MB
        fclose(f);
        return;
    }
    
    char *buffer = malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return;
    }
    
    size_t read = fread(buffer, 1, size, f);
    buffer[read] = '\0';
    fclose(f);
    
    // Parse for rule comments
    // Look for lines like: // SAFETY_RULE(name="...", context={...}, effect={...})
    const char *p = buffer;
    int line_num = 1;
    
    while (*p) {
        // Skip to next line
        const char *line_start = p;
        while (*p && *p != '\n') p++;
        if (*p == '\n') {
            p++;
            line_num++;
        }
        
        // Check if this line contains a rule comment
        const char *q = line_start;
        q = skip_whitespace(q);
        
        // Must start with // 
        if (q[0] != '/' || q[1] != '/') continue;
        q += 2;
        q = skip_whitespace(q);
        
        // Check for rule type
        if (strncmp(q, "SAFETY_RULE", 11) == 0 || 
            strncmp(q, "BEHAVIOR_RULE", 13) == 0 ||
            strncmp(q, "PATTERN_RULE", 12) == 0 ||
            strncmp(q, "RULE", 4) == 0) {
            
            // Extract rule type
            char rule_type[64] = {0};
            const char *type_end = q;
            while (*type_end && *type_end != '(' && !isspace(*type_end)) type_end++;
            size_t type_len = type_end - q;
            if (type_len >= sizeof(rule_type)) type_len = sizeof(rule_type) - 1;
            memcpy(rule_type, q, type_len);
            rule_type[type_len] = '\0';
            
            q = type_end;
            while (*q && *q != '(') q++; // Skip to opening paren
            if (*q != '(') continue;
            q++; // Skip '('
            
            ScannedRule rule = {0};
            strncpy(rule.rule_type, rule_type, sizeof(rule.rule_type) - 1);
            strncpy(rule.origin_file, file_path, sizeof(rule.origin_file) - 1);
            
            // Parse name="..."
            q = skip_whitespace(q);
            if (strncmp(q, "name=", 5) == 0) {
                q += 5;
                if (*q == '"') {
                    q++;
                    size_t name_i = 0;
                    while (*q && *q != '"' && name_i < sizeof(rule.name) - 1) {
                        rule.name[name_i++] = *q++;
                    }
                    rule.name[name_i] = '\0';
                    if (*q == '"') q++;
                }
            }
            
            // Parse context={...}
            q = skip_whitespace(q);
            if (*q == ',') q++;
            q = skip_whitespace(q);
            if (strncmp(q, "context={", 9) == 0) {
                q += 9;
                const char *context_start = q;
                int brace_depth = 1;
                while (*q && brace_depth > 0) {
                    if (*q == '{') brace_depth++;
                    else if (*q == '}') brace_depth--;
                    q++;
                }
                if (brace_depth == 0) {
                    size_t ctx_len = q - context_start - 1; // Exclude closing brace
                    if (ctx_len >= sizeof(rule.context)) ctx_len = sizeof(rule.context) - 1;
                    memcpy(rule.context, context_start, ctx_len);
                    rule.context[ctx_len] = '\0';
                }
            }
            
            // Parse effect={...}
            q = skip_whitespace(q);
            if (*q == ',') q++;
            q = skip_whitespace(q);
            if (strncmp(q, "effect={", 8) == 0) {
                q += 8;
                const char *effect_start = q;
                int brace_depth = 1;
                while (*q && brace_depth > 0) {
                    if (*q == '{') brace_depth++;
                    else if (*q == '}') brace_depth--;
                    q++;
                }
                if (brace_depth == 0) {
                    size_t eff_len = q - effect_start - 1; // Exclude closing brace
                    if (eff_len >= sizeof(rule.effect)) eff_len = sizeof(rule.effect) - 1;
                    memcpy(rule.effect, effect_start, eff_len);
                    rule.effect[eff_len] = '\0';
                }
            }
            
            // Emit rule to graph
            if (rule.name[0]) {
                printf("[mc_scaffold] Found rule: %s (%s)\n", rule.name, rule.rule_type);
                mc_scaffold_emit_rule(g, &rule);
            }
        }
    }
    
    free(buffer);
}

// Scan scaffolds directory (internal helper)
static void scan_scaffolds_dir(char ***found_files, size_t *found_count) {
    *found_count = 0;
    *found_files = NULL;
    size_t found_cap = 0;
    
    DIR *dir = opendir("scaffolds");
    if (!dir) {
        printf("[mc_scaffold] No scaffolds directory found\n");
        return;
    }
    
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        
        // Only process .c files
        size_t len = strlen(ent->d_name);
        if (len > 2 && strcmp(ent->d_name + len - 2, ".c") == 0) {
            char path[512];
            snprintf(path, sizeof(path), "scaffolds/%s", ent->d_name);
            
            if (*found_count >= found_cap) {
                found_cap = found_cap ? found_cap * 2 : 64;
                *found_files = realloc(*found_files, found_cap * sizeof(char*));
            }
            (*found_files)[(*found_count)++] = strdup(path);
        }
    }
    closedir(dir);
    
    printf("[mc_scaffold] Found %zu scaffold files\n", *found_count);
}

// MC function: Process all scaffold files (called from main loop)
void mc_process_scaffolds(Brain *g, uint64_t node_id) {
    static int processed = 0;
    
    // Check if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Check if scaffolds should be scanned
    if (!mc_scaffold_should_scan(g)) {
        printf("[mc_scaffold] Scaffolds already applied, skipping\n");
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        processed = 1;
        return;
    }
    
    if (processed) {
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        return;
    }
    
    printf("[mc_scaffold] Processing scaffold files...\n");
    
    // Scan for scaffold files
    char **found_files = NULL;
    size_t found_count = 0;
    scan_scaffolds_dir(&found_files, &found_count);
    
    if (found_count == 0) {
        printf("[mc_scaffold] No scaffold files found\n");
        // Mark as processed even if none found
        mark_scaffolds_applied(g);
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        processed = 1;
        return;
    }
    
    // Process each scaffold file
    int processed_any = 0;
    for (size_t i = 0; i < found_count; i++) {
        printf("[mc_scaffold] Processing: %s\n", found_files[i]);
        mc_scaffold_process_file(g, found_files[i]);
        processed_any = 1;
    }
    
    // Free file list
    for (size_t i = 0; i < found_count; i++) {
        free(found_files[i]);
    }
    free(found_files);
    
    if (processed_any) {
        // Count patterns created
        uint64_t pattern_count = 0;
        for (uint64_t i = 0; i < g->header->num_nodes; i++) {
            if (g->nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
                pattern_count++;
            }
        }
        printf("[mc_scaffold] Total pattern roots in graph: %llu\n", 
               (unsigned long long)pattern_count);
        
        // Mark scaffolds as applied
        mark_scaffolds_applied(g);
        
        // Cleanup: Delete scaffold files
        mc_scaffold_cleanup();
        
        printf("[mc_scaffold] Scaffold processing complete. Files deleted.\n");
        
        // Sync memory-mapped file to disk to persist patterns
        if (g->mmap_size > 0) {
            if (msync(g->header, g->mmap_size, MS_SYNC) == 0) {
                printf("[mc_scaffold] Synced %zu bytes to disk\n", g->mmap_size);
            } else {
                perror("[mc_scaffold] msync failed");
            }
        }
        
        // Create parse_c node in the graph (brain self-organizes)
        // Find the MC ID for parse_c
        uint32_t parse_mc_id = 0;
        for (uint32_t i = 0; i < g_mc_count; i++) {
            if (g_mc_table[i].name && strcmp(g_mc_table[i].name, "parse_c") == 0) {
                parse_mc_id = i;
                break;
            }
        }
        
        if (parse_mc_id > 0) {
            uint64_t parse_node = alloc_node(g);
            if (parse_node != UINT64_MAX && parse_node < g->header->num_nodes) {
                g->nodes[parse_node].kind = NODE_KIND_CONTROL;
                g->nodes[parse_node].mc_id = parse_mc_id;
                g->nodes[parse_node].bias = 5.0f; // Activate after scaffolds
                g->nodes[parse_node].a = 1.0f;
                printf("[mc_scaffold] Created parse_c node %llu in graph (brain self-organized)\n", 
                       (unsigned long long)parse_node);
            }
        }
    }
    
    processed = 1;
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

