#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

// Minimal token types
typedef enum {
    TOK_EOF = 0,
    TOK_IDENTIFIER,
    TOK_LPAREN,    // (
    TOK_RPAREN,    // )
    TOK_LBRACE,    // {
    TOK_RBRACE,    // }
    TOK_COMMA,     // ,
    TOK_SEMICOLON, // ;
    TOK_STAR,      // *
    TOK_OTHER
} TokenType;

typedef struct {
    TokenType type;
    char *text;
    size_t len;
    size_t line;
    size_t col;
} Token;

// Simple lexer state
typedef struct {
    const char *src;
    size_t pos;
    size_t len;
    size_t line;
    size_t col;
    Token current;
} Lexer;

// Function info for tracking
typedef struct FuncInfo {
    uint64_t func_node_id;
    char *name;
    size_t name_len;
    struct FuncInfo *next;
} FuncInfo;

static FuncInfo *g_functions = NULL;

// Tag node cache for semantic tagging
static uint64_t tag_function_id = UINT64_MAX;
static uint64_t tag_param_id = UINT64_MAX;
static uint64_t tag_call_id = UINT64_MAX;

// Helper: Allocate a node
static uint64_t alloc_node(Brain *g) {
    if (g->header->num_nodes >= g->header->node_cap) return UINT64_MAX;
    uint64_t id = g->header->num_nodes++;
    memset(&g->nodes[id], 0, sizeof(Node));
    return id;
}

// Helper: Add edge
static void add_edge(Brain *g, uint64_t src, uint64_t dst, float w, uint32_t flags) {
    if (g->header->num_edges >= g->header->edge_cap) return;
    Edge *e = &g->edges[g->header->num_edges++];
    e->src = src;
    e->dst = dst;
    e->w = w;
    e->flags = flags;
    e->usage_count = 1;
}

// Ensure a tag node exists (lazily create or return cached)
static uint64_t ensure_tag_node(Brain *g, uint64_t *cached_id, const char *label) {
    if (*cached_id != UINT64_MAX && *cached_id < g->header->num_nodes) {
        return *cached_id;
    }
    
    // Create a new tag node
    uint64_t tag_id = alloc_node(g);
    if (tag_id == UINT64_MAX) return UINT64_MAX;
    
    Node *tag = &g->nodes[tag_id];
    tag->kind = NODE_KIND_TAG;  // Use existing TAG node kind
    tag->a = 0.5f;
    tag->reliability = 1.0f;
    
    // Store label hash in value (for identification)
    uint32_t hash = 0;
    size_t len = strlen(label);
    for (size_t i = 0; i < len && i < 32; i++) {
        hash = hash * 31 + (unsigned char)label[i];
    }
    tag->value = (float)hash;
    
    // Link to a root META node if one exists (optional, for organization)
    // For now, just cache it
    *cached_id = tag_id;
    
    return tag_id;
}

// Helper: Create or find a string node (stores identifier text)
static uint64_t create_string_node(Brain *g, const char *text, size_t len) {
    // For now, create a DATA node and store hash in value
    // In a full system, we'd use blob storage
    uint64_t node_id = alloc_node(g);
    if (node_id == UINT64_MAX) return UINT64_MAX;
    
    Node *n = &g->nodes[node_id];
    n->kind = NODE_KIND_DATA;
    n->a = 0.5f;
    
    // Simple hash for identifier
    uint32_t hash = 0;
    for (size_t i = 0; i < len && i < 32; i++) {
        hash = hash * 31 + (unsigned char)text[i];
    }
    n->value = (float)hash;
    
    return node_id;
}

// Lexer functions
static void lexer_init(Lexer *l, const char *src, size_t len) {
    l->src = src;
    l->pos = 0;
    l->len = len;
    l->line = 1;
    l->col = 1;
    l->current.type = TOK_EOF;
    l->current.text = NULL;
    l->current.len = 0;
}

static int is_ident_char(char c) {
    return isalnum(c) || c == '_';
}

static void skip_whitespace(Lexer *l) {
    while (l->pos < l->len) {
        char c = l->src[l->pos];
        if (c == '\n') {
            l->line++;
            l->col = 1;
            l->pos++;
        } else if (isspace(c)) {
            l->col++;
            l->pos++;
        } else if (c == '/' && l->pos + 1 < l->len && l->src[l->pos + 1] == '/') {
            // Skip line comment
            while (l->pos < l->len && l->src[l->pos] != '\n') {
                l->pos++;
            }
        } else if (c == '/' && l->pos + 1 < l->len && l->src[l->pos + 1] == '*') {
            // Skip block comment
            l->pos += 2;
            while (l->pos + 1 < l->len) {
                if (l->src[l->pos] == '*' && l->src[l->pos + 1] == '/') {
                    l->pos += 2;
                    break;
                }
                if (l->src[l->pos] == '\n') {
                    l->line++;
                    l->col = 1;
                } else {
                    l->col++;
                }
                l->pos++;
            }
        } else {
            break;
        }
    }
}

static Token next_token(Lexer *l) {
    skip_whitespace(l);
    
    Token tok;
    tok.line = l->line;
    tok.col = l->col;
    tok.text = (char *)&l->src[l->pos];
    tok.len = 0;
    
    if (l->pos >= l->len) {
        tok.type = TOK_EOF;
        return tok;
    }
    
    char c = l->src[l->pos];
    
    if (is_ident_char(c)) {
        // Identifier
        tok.type = TOK_IDENTIFIER;
        while (l->pos < l->len && is_ident_char(l->src[l->pos])) {
            tok.len++;
            l->pos++;
            l->col++;
        }
    } else {
        tok.len = 1;
        l->pos++;
        l->col++;
        
        switch (c) {
            case '(': tok.type = TOK_LPAREN; break;
            case ')': tok.type = TOK_RPAREN; break;
            case '{': tok.type = TOK_LBRACE; break;
            case '}': tok.type = TOK_RBRACE; break;
            case ',': tok.type = TOK_COMMA; break;
            case ';': tok.type = TOK_SEMICOLON; break;
            case '*': tok.type = TOK_STAR; break;
            default: tok.type = TOK_OTHER; break;
        }
    }
    
    return tok;
}

// Register a function for call resolution
static void register_function(const char *name, size_t len, uint64_t func_node_id) {
    FuncInfo *fi = malloc(sizeof(FuncInfo));
    if (!fi) return;
    
    fi->name = malloc(len + 1);
    if (!fi->name) {
        free(fi);
        return;
    }
    memcpy(fi->name, name, len);
    fi->name[len] = '\0';
    fi->name_len = len;
    fi->func_node_id = func_node_id;
    fi->next = g_functions;
    g_functions = fi;
}

// Find function node by name
static uint64_t find_function_node(const char *name, size_t len) {
    FuncInfo *fi = g_functions;
    while (fi) {
        if (fi->name_len == len && memcmp(fi->name, name, len) == 0) {
            return fi->func_node_id;
        }
        fi = fi->next;
    }
    return UINT64_MAX;
}

// Parse a type (simplified - just skip until identifier)
static int parse_type(Lexer *l, Token *tok) {
    // Skip type keywords and qualifiers
    while (tok->type == TOK_IDENTIFIER || tok->type == TOK_STAR) {
        *tok = next_token(l);
        if (tok->type == TOK_LPAREN || tok->type == TOK_IDENTIFIER) {
            break;
        }
    }
    return 1;
}

// Parse parameter list
static void parse_params(Brain *g, Lexer *l, uint64_t func_node_id, Token *tok) {
    if (tok->type != TOK_LPAREN) return;
    *tok = next_token(l);
    
    while (tok->type != TOK_RPAREN && tok->type != TOK_EOF) {
        if (tok->type == TOK_RPAREN) break;
        
        // Skip type
        parse_type(l, tok);
        
        // Get parameter name
        if (tok->type == TOK_IDENTIFIER) {
            uint64_t param_node = alloc_node(g);
            if (param_node != UINT64_MAX) {
                Node *pn = &g->nodes[param_node];
                pn->kind = NODE_KIND_DATA;  // Generic node kind
                pn->a = 0.3f;
                
                // Create string node for param name
                uint64_t name_node = create_string_node(g, tok->text, tok->len);
                if (name_node != UINT64_MAX) {
                    add_edge(g, param_node, name_node, 1.0f, EDGE_FLAG_BIND);
                }
                
                // Tag it as a parameter
                uint64_t tag_param = ensure_tag_node(g, &tag_param_id, "TAG_PARAM");
                if (tag_param != UINT64_MAX) {
                    add_edge(g, tag_param, param_node, 1.0f, EDGE_FLAG_BIND);
                }
                
                // Link param to function using BIND (semantic relationship)
                add_edge(g, func_node_id, param_node, 1.0f, EDGE_FLAG_BIND);
            }
        }
        
        *tok = next_token(l);
        
        if (tok->type == TOK_COMMA) {
            *tok = next_token(l);
        } else if (tok->type != TOK_RPAREN) {
            // Skip to next comma or closing paren
            while (tok->type != TOK_COMMA && tok->type != TOK_RPAREN && tok->type != TOK_EOF) {
                *tok = next_token(l);
            }
            if (tok->type == TOK_COMMA) {
                *tok = next_token(l);
            }
        }
    }
    
    if (tok->type == TOK_RPAREN) {
        *tok = next_token(l);
    }
}

// Parse function body and detect calls
static void parse_body(Brain *g, Lexer *l, uint64_t func_node_id, Token *tok) {
    if (tok->type != TOK_LBRACE) return;
    
    int brace_depth = 1;
    *tok = next_token(l);
    
    while (brace_depth > 0 && tok->type != TOK_EOF) {
        if (tok->type == TOK_LBRACE) {
            brace_depth++;
        } else if (tok->type == TOK_RBRACE) {
            brace_depth--;
            if (brace_depth == 0) {
                *tok = next_token(l);
                return;
            }
        } else if (tok->type == TOK_IDENTIFIER) {
            // Might be a function call
            Token ident = *tok;
            *tok = next_token(l);
            
            if (tok->type == TOK_LPAREN) {
                // This is a function call
                uint64_t call_node = alloc_node(g);
                if (call_node != UINT64_MAX) {
                    Node *cn = &g->nodes[call_node];
                    cn->kind = NODE_KIND_DATA;  // Generic node kind
                    cn->a = 0.4f;
                    
                    // Create string node for call name
                    uint64_t name_node = create_string_node(g, ident.text, ident.len);
                    if (name_node != UINT64_MAX) {
                        add_edge(g, call_node, name_node, 1.0f, EDGE_FLAG_BIND);
                    }
                    
                    // Tag it as a call
                    uint64_t tag_call = ensure_tag_node(g, &tag_call_id, "TAG_CALL");
                    if (tag_call != UINT64_MAX) {
                        add_edge(g, tag_call, call_node, 1.0f, EDGE_FLAG_BIND);
                    }
                    
                    // Link call to containing function (using BIND for semantic relationship)
                    add_edge(g, func_node_id, call_node, 1.0f, EDGE_FLAG_BIND);
                    
                    // Try to link to target function if known (using BIND for semantic relationship)
                    uint64_t target_func = find_function_node(ident.text, ident.len);
                    if (target_func != UINT64_MAX) {
                        add_edge(g, call_node, target_func, 1.0f, EDGE_FLAG_BIND);
                    }
                }
                
                // Skip arguments
                int paren_depth = 1;
                *tok = next_token(l);
                while (paren_depth > 0 && tok->type != TOK_EOF) {
                    if (tok->type == TOK_LPAREN) paren_depth++;
                    else if (tok->type == TOK_RPAREN) paren_depth--;
                    *tok = next_token(l);
                }
                continue;
            }
        }
        
        *tok = next_token(l);
    }
}

// Main parser
static void parse_c_file(Brain *g, const char *src, size_t len) {
    Lexer l;
    lexer_init(&l, src, len);
    
    Token tok = next_token(&l);
    
    while (tok.type != TOK_EOF) {
        // Look for function definition: type identifier ( params ) { body }
        if (tok.type == TOK_IDENTIFIER) {
            // Might be return type or function name
            tok = next_token(&l);
            
            // Skip type and qualifiers
            parse_type(&l, &tok);
            
            // Now we should have function name
            if (tok.type == TOK_IDENTIFIER) {
                Token name_tok = tok;
                tok = next_token(&l);
                
                if (tok.type == TOK_LPAREN) {
                    // This is a function definition!
                    uint64_t func_node = alloc_node(g);
                    if (func_node != UINT64_MAX) {
                        Node *fn = &g->nodes[func_node];
                        fn->kind = NODE_KIND_DATA;  // Generic node kind
                        fn->a = 0.6f;
                        fn->reliability = 0.8f;
                        
                        // Create string node for function name
                        uint64_t name_node = create_string_node(g, name_tok.text, name_tok.len);
                        if (name_node != UINT64_MAX) {
                            add_edge(g, func_node, name_node, 1.0f, EDGE_FLAG_BIND);
                        }
                        
                        // Tag it as a function
                        uint64_t tag_fn = ensure_tag_node(g, &tag_function_id, "TAG_FUNCTION");
                        if (tag_fn != UINT64_MAX) {
                            add_edge(g, tag_fn, func_node, 1.0f, EDGE_FLAG_BIND);
                        }
                        
                        // Register function
                        register_function(name_tok.text, name_tok.len, func_node);
                        
                        // Parse parameters
                        parse_params(g, &l, func_node, &tok);
                        
                        // Parse body (and detect calls)
                        if (tok.type == TOK_LBRACE) {
                            parse_body(g, &l, func_node, &tok);
                        }
                        
                        fprintf(stderr, "[mc_parse] Found function: %.*s (node %llu)\n",
                                (int)name_tok.len, name_tok.text, (unsigned long long)func_node);
                    }
                    continue;
                }
            }
        }
        
        tok = next_token(&l);
    }
}

// Static list to track parsed files (to avoid re-parsing)
static char **parsed_files = NULL;
static size_t parsed_count = 0;
static size_t parsed_cap = 0;

static int is_file_parsed(const char *path) {
    for (size_t i = 0; i < parsed_count; i++) {
        if (strcmp(parsed_files[i], path) == 0) {
            return 1;
        }
    }
    return 0;
}

static void mark_file_parsed(const char *path) {
    if (is_file_parsed(path)) return;
    
    if (parsed_count >= parsed_cap) {
        parsed_cap = parsed_cap ? parsed_cap * 2 : 64;
        parsed_files = realloc(parsed_files, parsed_cap * sizeof(char*));
    }
    parsed_files[parsed_count++] = strdup(path);
}

// Recursively find all .c files in directory
static void find_c_files_recursive(const char *dir_path, char ***found_files, size_t *found_count, size_t *found_cap) {
    DIR *dir = opendir(dir_path);
    if (!dir) return;
    
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        
        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name);
        
        struct stat st;
        if (stat(path, &st) == 0) {
            if (S_ISREG(st.st_mode)) {
                size_t len = strlen(path);
                // Accept .c, .cpp, .h, .hpp files
                if ((len > 2 && strcmp(path + len - 2, ".c") == 0) ||
                    (len > 4 && strcmp(path + len - 4, ".cpp") == 0) ||
                    (len > 2 && strcmp(path + len - 2, ".h") == 0) ||
                    (len > 4 && strcmp(path + len - 4, ".hpp") == 0)) {
                    // Add to found_files list
                    if (*found_count >= *found_cap) {
                        *found_cap = *found_cap ? *found_cap * 2 : 128;
                        *found_files = realloc(*found_files, *found_cap * sizeof(char*));
                    }
                    (*found_files)[(*found_count)++] = strdup(path);
                }
            } else if (S_ISDIR(st.st_mode)) {
                // Skip .git, node_modules, build, and scaffolds
                if (strcmp(ent->d_name, ".git") != 0 && 
                    strcmp(ent->d_name, "node_modules") != 0 &&
                    strcmp(ent->d_name, "build") != 0 &&
                    strcmp(ent->d_name, "scaffolds") != 0) {
                    find_c_files_recursive(path, found_files, found_count, found_cap);
                }
            }
        }
    }
    closedir(dir);
}

// Get file path from connected nodes or scan for .c files
static int get_file_path(Brain *g, uint64_t node_id, char *path, size_t path_size) {
    // First, try to get path from connected nodes
    for (uint64_t i = 0; i < g->header->num_edges; i++) {
        Edge *e = &g->edges[i];
        if (e->src == node_id && g->nodes[e->dst].kind == NODE_KIND_DATA) {
            // In a full system, we'd decode path from node data
            // For now, we'll use scanning
        }
    }
    
    // If no path from graph, scan for .c files
    return 0;
}

// Helper: Check if new repos need to be parsed (called before parsing)
static void check_for_new_repos(void) {
    // Check if mc_git_auto_learn exists and trigger it
    // This allows git clone to happen automatically when parse runs
    extern void mc_git_auto_learn(Brain *, uint64_t);
    // We'll call it via a special node if it exists - for now just check file
    FILE *f = fopen("github_urls.txt", "r");
    if (f) {
        // File exists - it will be processed by git_auto_learn when that node activates
        fclose(f);
    }
}

// MC function: Parse C file - now discovers and parses ALL .c files
void mc_parse_c_file(Brain *g, uint64_t node_id) {
    // Check for new repositories to clone before parsing
    check_for_new_repos();
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Find all .c files in current directory and plugins/
    char **found_files = NULL;
    size_t found_count = 0;
    size_t found_cap = 0;
    
    find_c_files_recursive(".", &found_files, &found_count, &found_cap);
    find_c_files_recursive("./plugins", &found_files, &found_count, &found_cap);
    find_c_files_recursive("./ingested_repos", &found_files, &found_count, &found_cap);
    
    // Skip scaffolds directory - those are handled separately
    
    if (found_count == 0) {
        fprintf(stderr, "[mc_parse] No .c files found to parse.\n");
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        return;
    }
    
    fprintf(stderr, "[mc_parse] Found %zu .c files. Parsing...\n", found_count);
    
    int parsed_any = 0;
    
    // Parse each .c file that hasn't been parsed yet
    for (size_t i = 0; i < found_count; i++) {
        const char *file_path = found_files[i];
        
        if (is_file_parsed(file_path)) {
            continue; // Skip already parsed files
        }
        
        FILE *f = fopen(file_path, "r");
        if (!f) {
            continue;
        }
        
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        
        if (size > 0 && size < 10 * 1024 * 1024) { // Max 10MB
            char *buffer = malloc(size + 1);
            if (buffer) {
                size_t read = fread(buffer, 1, size, f);
                buffer[read] = '\0';
                
                fprintf(stderr, "[mc_parse] Parsing %s (%ld bytes)...\n", file_path, size);
                
                // Create a file node in the graph to represent this file
                uint64_t file_node = alloc_node(g);
                if (file_node != UINT64_MAX) {
                    Node *fn = &g->nodes[file_node];
                    fn->kind = NODE_KIND_DATA;
                    fn->a = 0.7f;
                    
                    // Create string node for filename
                    const char *basename = strrchr(file_path, '/');
                    if (!basename) basename = file_path;
                    else basename++;
                    
                    uint64_t name_node = create_string_node(g, basename, strlen(basename));
                    if (name_node != UINT64_MAX) {
                        add_edge(g, file_node, name_node, 1.0f, EDGE_FLAG_BIND);
                    }
                    
                    // Link to parse node
                    add_edge(g, node_id, file_node, 1.0f, EDGE_FLAG_CONTROL);
                }
                
                // Parse the C file content into the graph
                parse_c_file(g, buffer, read);
                
                // Mark as parsed
                mark_file_parsed(file_path);
                parsed_any = 1;
                
                free(buffer);
            }
        }
        fclose(f);
    }
    
    // Free found_files list
    for (size_t i = 0; i < found_count; i++) {
        free(found_files[i]);
    }
    free(found_files);
    
    if (parsed_any) {
        // Create PARSE_OK node
        uint64_t ok_node = alloc_node(g);
        if (ok_node != UINT64_MAX) {
            Node *n = &g->nodes[ok_node];
            n->kind = NODE_KIND_META;
            n->a = 1.0f;
            n->value = 0x50415253; // "PARS"
            add_edge(g, node_id, ok_node, 1.0f, EDGE_FLAG_CONTROL);
        }
        
        fprintf(stderr, "[mc_parse] Parse complete. Functions registered in graph.\n");
    }
    
    // Keep parse node slightly active so it can re-trigger for new files
    // Instead of fully deactivating, reduce bias but keep checking
    if (parsed_any) {
        g->nodes[node_id].bias = 0.1f; // Low bias - can reactivate easily
    } else {
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
    }
}

