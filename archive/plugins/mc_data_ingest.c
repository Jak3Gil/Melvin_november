#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <dirent.h>

// External helpers (from melvin.c)
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

// Data ingestion state
static char **data_files = NULL;
static size_t data_file_count = 0;
static size_t data_file_cap = 0;
static size_t current_file_idx = 0;
static FILE *current_fp = NULL;
static uint64_t current_file_node = UINT64_MAX;

// Scan for data files
static void scan_data_directory(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) return;
    
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name);
        
        struct stat st;
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
            // Check file extension
            size_t len = strlen(path);
            if (len > 4) {
                const char *ext = path + len - 4;
                // Accept common data formats
                if (strcmp(ext, ".txt") == 0 || strcmp(ext, ".html") == 0 ||
                    strcmp(ext, ".json") == 0 || strcmp(ext, ".xml") == 0 ||
                    strcmp(ext, ".csv") == 0 || strcmp(ext, ".tsv") == 0) {
                    // Add to list
                    if (data_file_count >= data_file_cap) {
                        data_file_cap = data_file_cap ? data_file_cap * 2 : 1024;
                        data_files = realloc(data_files, data_file_cap * sizeof(char*));
                    }
                    data_files[data_file_count++] = strdup(path);
                }
            }
        } else if (S_ISDIR(st.st_mode)) {
            // Recursively scan subdirectories
            scan_data_directory(path);
        }
    }
    closedir(dir);
}

// MC function: Ingest data files (CommonCrawl, text datasets, etc.)
void mc_data_ingest(Brain *g, uint64_t node_id) {
    static int scanned = 0;
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Scan for data files on first activation
    if (!scanned) {
        printf("[mc_data_ingest] Scanning for data files...\n");
        scan_data_directory("data");
        scan_data_directory("corpus");
        scan_data_directory("ingested_repos");
        printf("[mc_data_ingest] Found %zu data files\n", data_file_count);
        scanned = 1;
    }
    
    if (data_file_count == 0) {
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    // Process files one at a time
    if (!current_fp && current_file_idx < data_file_count) {
        const char *file_path = data_files[current_file_idx];
        current_fp = fopen(file_path, "rb");
        if (current_fp) {
            printf("[mc_data_ingest] Processing: %s\n", file_path);
            
            // Create file node
            current_file_node = alloc_node(g);
            if (current_file_node != UINT64_MAX) {
                Node *fn = &g->nodes[current_file_node];
                fn->kind = NODE_KIND_DATA;
                fn->a = 0.8f;
                
                // Hash filename
                uint32_t hash = 0;
                size_t len = strlen(file_path);
                for (size_t i = 0; i < len && i < 32; i++) {
                    hash = hash * 31 + (unsigned char)file_path[i];
                }
                fn->value = (float)hash;
                
                add_edge(g, node_id, current_file_node, 1.0f, EDGE_FLAG_CONTROL);
            }
        } else {
            current_file_idx++;
            return;
        }
    }
    
    if (!current_fp) {
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    // Read chunk of data
    uint8_t buffer[4096];
    size_t n = fread(buffer, 1, sizeof(buffer), current_fp);
    
    if (n > 0) {
        // Process bytes: create nodes and sequence edges
        uint64_t prev_byte_node = UINT64_MAX;
        
        for (size_t i = 0; i < n; i++) {
            uint8_t byte = buffer[i];
            
            // Activate byte node (first 256 nodes are byte nodes)
            if (byte < g->header->num_nodes) {
                Node *byte_node = &g->nodes[byte];
                byte_node->a = 0.7f;
                byte_node->value = (float)byte;
                byte_node->kind = NODE_KIND_DATA;
                
                // Link to file node
                if (current_file_node != UINT64_MAX) {
                    add_edge(g, current_file_node, byte, 1.0f, EDGE_FLAG_BIND);
                }
                
                // Create sequence edge from previous byte
                if (prev_byte_node != UINT64_MAX && prev_byte_node < g->header->num_nodes) {
                    add_edge(g, prev_byte_node, byte, 1.0f, EDGE_FLAG_SEQ);
                }
                
                prev_byte_node = byte;
            }
        }
    }
    
    // Check if file is done
    if (n < sizeof(buffer)) {
        fclose(current_fp);
        current_fp = NULL;
        current_file_node = UINT64_MAX;
        current_file_idx++;
        
        if (current_file_idx >= data_file_count) {
            // All files processed
            printf("[mc_data_ingest] Finished processing all data files\n");
            g->nodes[node_id].a = 0.0f;
            g->nodes[node_id].bias = -5.0f;
        }
    }
}

// MC function: Process video data (frame extraction)
void mc_video_ingest(Brain *g, uint64_t node_id) {
    // Placeholder for video processing
    // In full implementation, would use ffmpeg or similar to extract frames
    // For now, just acknowledge video files exist
    
    static int scanned = 0;
    
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    if (!scanned) {
        printf("[mc_video_ingest] Video processing requires frame extraction\n");
        printf("[mc_video_ingest] Place video files in data/video/ directory\n");
        scanned = 1;
    }
    
    // TODO: Extract frames, create frame nodes, temporal edges
    // This would require video processing library (ffmpeg, OpenCV, etc.)
    
    g->nodes[node_id].a = 0.0f;
}

// MC function: Process audio data (waveform processing)
void mc_audio_ingest(Brain *g, uint64_t node_id) {
    // Placeholder for audio processing
    // In full implementation, would extract audio features
    
    static int scanned = 0;
    
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    if (!scanned) {
        printf("[mc_audio_ingest] Audio processing requires feature extraction\n");
        printf("[mc_audio_ingest] Place audio files in data/audio/ directory\n");
        scanned = 1;
    }
    
    // TODO: Extract audio features, create feature nodes, temporal edges
    // This would require audio processing library (libsndfile, etc.)
    
    g->nodes[node_id].a = 0.0f;
}

