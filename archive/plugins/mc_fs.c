#define _POSIX_C_SOURCE 200809L
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

// External helpers (defined in melvin.c)
// Note: These are resolved at runtime when plugin is loaded
extern uint64_t alloc_node(Brain *);
// add_edge is not available to plugins - we'll skip edge creation for now
// or plugins can implement their own edge creation logic

// Static state for FS exploration
static char **file_list = NULL;
static size_t file_count = 0;
static size_t file_cap = 0;
static size_t current_file_idx = 0;
static FILE *current_fp = NULL;

static void add_file_to_list(const char *path) {
    if (file_count >= file_cap) {
        file_cap = file_cap ? file_cap * 2 : 1024;
        file_list = realloc(file_list, file_cap * sizeof(char*));
    }
    file_list[file_count++] = strdup(path);
}

static void scan_dir_recursive(const char *base_path) {
    DIR *dir = opendir(base_path);
    if (!dir) return;

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", base_path, ent->d_name);
        
        struct stat st;
        if (stat(path, &st) == 0) {
            if (S_ISREG(st.st_mode)) {
                add_file_to_list(path);
            } else if (S_ISDIR(st.st_mode)) {
                scan_dir_recursive(path);
            }
        }
    }
    closedir(dir);
}

void mc_fs_seed(Brain *g, uint64_t node_id) {
    static int seeded = 0;
    if (seeded) return;

    printf("[MC] Seeding filesystem...\n");
    scan_dir_recursive("."); // Scan current dir for now, or "./corpus" if exists
    
    // Also try ./data and ./corpus specifically if they exist
    scan_dir_recursive("./data");
    scan_dir_recursive("./corpus");

    printf("[MC] Found %zu files.\n", file_count);

    // Create nodes for files (optional, or just track internally)
    // For now, we just have the list.
    
    seeded = 1;
}

// Scan scaffolds directory and return list of scaffold files
void mc_fs_scan_scaffolds(char ***found_files, size_t *found_count) {
    *found_count = 0;
    *found_files = NULL;
    size_t found_cap = 0;
    
    DIR *dir = opendir("scaffolds");
    if (!dir) {
        printf("[MC] No scaffolds directory found\n");
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
    
    printf("[MC] Found %zu scaffold files\n", *found_count);
}

void mc_fs_read_chunk(Brain *g, uint64_t node_id) {
    if (file_count == 0) return;
    if (current_file_idx >= file_count) current_file_idx = 0; // Loop or stop? Loop for now.

    if (!current_fp) {
        const char *path = file_list[current_file_idx];
        current_fp = fopen(path, "rb");
        if (!current_fp) {
            // Failed to open, skip
            current_file_idx++;
            return;
        }
        printf("[MC] Reading file: %s\n", path);
        // Create a node representing this file if we want
    }

    uint8_t buffer[4096];
    size_t n = fread(buffer, 1, sizeof(buffer), current_fp);
    
    if (n > 0) {
        static uint64_t last_byte_node = 0;
        
        for (size_t i = 0; i < n; i++) {
            uint8_t b = buffer[i];
            // Activate node 0-255
            if (b < g->header->num_nodes) { // Assuming first 256 nodes are bytes
                 g->nodes[b].a = 1.0f;
                 g->nodes[b].value = (float)b;
                 g->nodes[b].kind = NODE_KIND_DATA;
                 
                 // Note: Edge creation skipped - plugins don't have direct access to add_edge
                 // The graph can learn sequences from activation patterns instead
                 last_byte_node = b;
            }
        }
    }

    if (n < sizeof(buffer)) {
        // EOF or Error
        fclose(current_fp);
        current_fp = NULL;
        current_file_idx++;
    }
}

