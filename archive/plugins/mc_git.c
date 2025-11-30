#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>

// External helpers (from melvin.c)
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

// Track cloned repositories
static char **cloned_repos = NULL;
static size_t cloned_count = 0;
static size_t cloned_cap = 0;

// Track pending URLs to clone
static char **pending_urls = NULL;
static size_t pending_count = 0;
static size_t pending_cap = 0;

// Check if URL is already cloned
static int is_repo_cloned(const char *url) {
    for (size_t i = 0; i < cloned_count; i++) {
        if (strcmp(cloned_repos[i], url) == 0) {
            return 1;
        }
    }
    return 0;
}

// Extract repo name from URL
static void extract_repo_name(const char *url, char *name_out, size_t name_size) {
    // Extract from github.com/user/repo.git or github.com/user/repo
    const char *start = strstr(url, "github.com/");
    if (!start) {
        snprintf(name_out, name_size, "unknown_repo");
        return;
    }
    start += 11; // Skip "github.com/"
    
    const char *end = strchr(start, '/');
    if (end) {
        end++;
        const char *end2 = strchr(end, '.');
        if (end2) {
            size_t len = end2 - end;
            if (len >= name_size) len = name_size - 1;
            memcpy(name_out, end, len);
            name_out[len] = '\0';
        } else {
            snprintf(name_out, name_size, "%s", end);
        }
    } else {
        snprintf(name_out, name_size, "unknown_repo");
    }
    
    // Clean up the name
    for (char *p = name_out; *p; p++) {
        if (!isalnum(*p) && *p != '_' && *p != '-') {
            *p = '_';
        }
    }
}

// Add URL to pending queue
static void add_pending_url(const char *url) {
    if (!url || strlen(url) == 0) return;
    
    // Check if already pending
    for (size_t i = 0; i < pending_count; i++) {
        if (strcmp(pending_urls[i], url) == 0) {
            return; // Already pending
        }
    }
    
    // Check if already cloned
    if (is_repo_cloned(url)) {
        printf("[mc_git] Repository already cloned: %s\n", url);
        return;
    }
    
    if (pending_count >= pending_cap) {
        pending_cap = pending_cap ? pending_cap * 2 : 64;
        pending_urls = realloc(pending_urls, pending_cap * sizeof(char*));
    }
    pending_urls[pending_count++] = strdup(url);
    printf("[mc_git] Added to queue: %s\n", url);
}

// Monitor stdin for GitHub URLs (non-blocking)
static void check_stdin_for_urls(void) {
    static char line_buffer[4096];
    static size_t buffer_pos = 0;
    
    // Non-blocking read from stdin
    char c;
    while (read(0, &c, 1) == 1) {
        if (c == '\n' || c == '\r') {
            if (buffer_pos > 0) {
                line_buffer[buffer_pos] = '\0';
                
                // Check if line contains github.com URL
                if (strstr(line_buffer, "github.com") != NULL) {
                    // Extract URL (could be full URL or just the part)
                    char url[512] = {0};
                    const char *start = strstr(line_buffer, "http");
                    if (start) {
                        const char *end = start;
                        while (*end && *end != ' ' && *end != '\n' && *end != '\r' && *end != '\t') {
                            end++;
                        }
                        size_t len = end - start;
                        if (len >= sizeof(url)) len = sizeof(url) - 1;
                        memcpy(url, start, len);
                        url[len] = '\0';
                        
                        // Ensure it ends with .git or add it
                        if (!strstr(url, ".git") && url[strlen(url)-1] != '/') {
                            strcat(url, ".git");
                        }
                        
                        add_pending_url(url);
                    } else {
                        // Try without http prefix
                        start = strstr(line_buffer, "github.com");
                        if (start) {
                            snprintf(url, sizeof(url), "https://%s", start);
                            const char *end = url;
                            while (*end && *end != ' ' && *end != '\n' && *end != '\r') {
                                end++;
                            }
                            size_t len = end - url;
                            url[len] = '\0';
                            if (!strstr(url, ".git") && url[strlen(url)-1] != '/') {
                                strcat(url, ".git");
                            }
                            add_pending_url(url);
                        }
                    }
                }
                
                buffer_pos = 0;
            }
        } else if (buffer_pos < sizeof(line_buffer) - 1) {
            line_buffer[buffer_pos++] = c;
        }
    }
}

// Monitor file for GitHub URLs
static void check_file_for_urls(void) {
    FILE *f = fopen("github_urls.txt", "r");
    if (!f) return;
    
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        // Trim whitespace
        size_t len = strlen(line);
        while (len > 0 && isspace(line[len-1])) {
            line[--len] = '\0';
        }
        
        if (len == 0) continue;
        
        // Check if line starts with # (comment)
        if (line[0] == '#') continue;
        
        // Check if contains github.com
        if (strstr(line, "github.com") != NULL) {
            char url[512] = {0};
            
            // Extract URL
            const char *start = strstr(line, "http");
            if (start) {
                const char *end = start;
                while (*end && *end != ' ' && *end != '\n' && *end != '\r' && *end != '\t' && *end != '#') {
                    end++;
                }
                size_t url_len = end - start;
                if (url_len >= sizeof(url)) url_len = sizeof(url) - 1;
                memcpy(url, start, url_len);
                url[url_len] = '\0';
                
                // Ensure ends with .git
                if (!strstr(url, ".git") && url[strlen(url)-1] != '/') {
                    strcat(url, ".git");
                }
                
                add_pending_url(url);
            } else {
                start = strstr(line, "github.com");
                if (start) {
                    snprintf(url, sizeof(url), "https://%s", start);
                    char *end = url + strlen(url);
                    while (end > url && (*end == ' ' || *end == '\n' || *end == '\r' || *end == '\t' || *end == '#')) {
                        *end = '\0';
                        end--;
                    }
                    if (!strstr(url, ".git") && url[strlen(url)-1] != '/') {
                        strcat(url, ".git");
                    }
                    add_pending_url(url);
                }
            }
        }
    }
    fclose(f);
}

// Clone a single repository
static int clone_repository(const char *repo_url) {
    if (is_repo_cloned(repo_url)) {
        printf("[mc_git] Already cloned: %s\n", repo_url);
        return 0;
    }
    
    char repo_name[256];
    extract_repo_name(repo_url, repo_name, sizeof(repo_name));
    
    char target_dir[512];
    snprintf(target_dir, sizeof(target_dir), "ingested_repos/%s", repo_name);
    
    // Check if directory already exists
    struct stat st;
    if (stat(target_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
        printf("[mc_git] Directory already exists: %s\n", target_dir);
        // Add to cloned list
        if (cloned_count >= cloned_cap) {
            cloned_cap = cloned_cap ? cloned_cap * 2 : 64;
            cloned_repos = realloc(cloned_repos, cloned_cap * sizeof(char*));
        }
        cloned_repos[cloned_count++] = strdup(repo_url);
        return 0;
    }
    
    // Create target directory
    system("mkdir -p ingested_repos");
    
    printf("[mc_git] 🚀 Cloning: %s -> %s\n", repo_url, target_dir);
    
    // Build git clone command
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "git clone --quiet %s %s 2>&1", repo_url, target_dir);
    
    int rc = system(cmd);
    
    if (rc == 0) {
        printf("[mc_git] ✅ Cloned successfully: %s\n", repo_name);
        
        // Add to cloned list
        if (cloned_count >= cloned_cap) {
            cloned_cap = cloned_cap ? cloned_cap * 2 : 64;
            cloned_repos = realloc(cloned_repos, cloned_cap * sizeof(char*));
        }
        cloned_repos[cloned_count++] = strdup(repo_url);
        
        return 1; // Success
    } else {
        fprintf(stderr, "[mc_git] ❌ Clone failed: %s (exit code %d)\n", repo_url, rc);
        return 0;
    }
}

// MC function: Auto-learn from GitHub URLs
void mc_git_auto_learn(Brain *g, uint64_t node_id) {
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Check stdin for URLs (non-blocking)
    check_stdin_for_urls();
    
    // Check file for URLs (github_urls.txt)
    check_file_for_urls();
    
    // Process pending URLs (one at a time per tick)
    if (pending_count > 0) {
        char *url = pending_urls[0];
        
        printf("[mc_git] Processing queued URL: %s\n", url);
        
        if (clone_repository(url)) {
            // Successfully cloned - create success node
            uint64_t success_node = alloc_node(g);
            if (success_node != UINT64_MAX) {
                Node *sn = &g->nodes[success_node];
                sn->kind = NODE_KIND_META;
                sn->a = 1.0f;
                sn->value = 0x47495443; // "GITC"
                
                // Store URL hash
                uint32_t hash = 0;
                size_t len = strlen(url);
                for (size_t i = 0; i < len && i < 32; i++) {
                    hash = hash * 31 + (unsigned char)url[i];
                }
                sn->flags = hash & 0xFFFF;
                
                add_edge(g, node_id, success_node, 1.0f, EDGE_FLAG_CONTROL);
                
                // Trigger parse node activation (by creating edge to parse node)
                // Find parse node or create signal node
                printf("[mc_git] Repository cloned - ready for parsing\n");
            }
        }
        
        // Remove from pending queue
        free(pending_urls[0]);
        for (size_t i = 1; i < pending_count; i++) {
            pending_urls[i-1] = pending_urls[i];
        }
        pending_count--;
    }
    
    // Keep node active while there's work to do
    if (pending_count == 0) {
        // No pending work, but keep checking
        // Don't deactivate - let it stay active to monitor for new URLs
    }
}

// MC function: Clone a specific repository (manual trigger)
void mc_git_clone(Brain *g, uint64_t node_id) {
    static int cloned = 0;
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    if (cloned) {
        g->nodes[node_id].a = 0.0f;
        g->nodes[node_id].bias = -5.0f;
        return;
    }
    
    // Default repository
    const char *repo_url = "https://github.com/Jak3Gil/melvin-unified-brain.git";
    
    if (clone_repository(repo_url)) {
        uint64_t success_node = alloc_node(g);
        if (success_node != UINT64_MAX) {
            Node *cn = &g->nodes[success_node];
            cn->kind = NODE_KIND_META;
            cn->a = 1.0f;
            cn->value = 0x47495443; // "GITC"
            add_edge(g, node_id, success_node, 1.0f, EDGE_FLAG_CONTROL);
        }
    }
    
    cloned = 1;
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}
