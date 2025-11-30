/*
 * melvin_pack_corpus.c - Corpus Packing Tool
 * 
 * Builds a v2 melvin.m file with a huge cold-data slab containing
 * all files from a directory tree. No semantic processing - just
 * raw byte concatenation.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>

/* Default hot region sizes */
#define DEFAULT_HOT_NODES 1000
#define DEFAULT_HOT_EDGES 10000
#define DEFAULT_HOT_BLOB_BYTES (1024 * 1024)  /* 1 MB */

/* Corpus file entry */
typedef struct {
    char     *path;        /* Full path (malloc'd) */
    uint64_t  size;        /* File size in bytes */
} CorpusFile;

/* Dynamic array for corpus files */
typedef struct {
    CorpusFile *files;
    size_t       count;
    size_t       capacity;
} CorpusFileList;

static CorpusFileList corpus_list = {NULL, 0, 0};

/* Add file to corpus list */
static int corpus_add_file(const char *path, uint64_t size) {
    if (corpus_list.count >= corpus_list.capacity) {
        size_t new_cap = corpus_list.capacity == 0 ? 64 : corpus_list.capacity * 2;
        CorpusFile *new_files = realloc(corpus_list.files, new_cap * sizeof(CorpusFile));
        if (!new_files) return -1;
        corpus_list.files = new_files;
        corpus_list.capacity = new_cap;
    }
    
    corpus_list.files[corpus_list.count].path = strdup(path);
    corpus_list.files[corpus_list.count].size = size;
    if (!corpus_list.files[corpus_list.count].path) return -1;
    
    corpus_list.count++;
    return 0;
}

/* Free corpus list */
static void corpus_free(void) {
    for (size_t i = 0; i < corpus_list.count; i++) {
        free(corpus_list.files[i].path);
    }
    free(corpus_list.files);
    corpus_list.files = NULL;
    corpus_list.count = 0;
    corpus_list.capacity = 0;
}

/* Recursively walk directory and collect files (Pass 1: size computation) */
static int walk_directory(const char *dir_path, const char *base_path, uint64_t *total_bytes) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "Error: Cannot open directory '%s': %s\n", dir_path, strerror(errno));
        return -1;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        /* Build full path */
        char full_path[PATH_MAX];
        int ret = snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
        if (ret < 0 || ret >= (int)sizeof(full_path)) {
            fprintf(stderr, "Warning: Path too long, skipping: %s/%s\n", dir_path, entry->d_name);
            continue;
        }
        
        struct stat st;
        if (lstat(full_path, &st) < 0) {
            fprintf(stderr, "Warning: Cannot stat '%s': %s\n", full_path, strerror(errno));
            continue;
        }
        
        /* Only process regular files */
        if (S_ISREG(st.st_mode)) {
            /* Store FULL path (needed for Pass 2 streaming) */
            uint64_t file_size = (uint64_t)st.st_size;
            if (corpus_add_file(full_path, file_size) < 0) {
                closedir(dir);
                return -1;
            }
            
            *total_bytes += file_size;
            fprintf(stdout, "  %s (%llu bytes)\n", full_path, (unsigned long long)file_size);
            fflush(stdout);
        } else if (S_ISDIR(st.st_mode)) {
            /* Recurse into subdirectory */
            char new_base[PATH_MAX];
            if (base_path) {
                snprintf(new_base, sizeof(new_base), "%s/%s", base_path, entry->d_name);
            } else {
                strncpy(new_base, entry->d_name, sizeof(new_base) - 1);
                new_base[sizeof(new_base) - 1] = '\0';
            }
            
            if (walk_directory(full_path, new_base, total_bytes) < 0) {
                closedir(dir);
                return -1;
            }
        }
        /* Ignore symlinks, sockets, etc. */
    }
    
    closedir(dir);
    return 0;
}

/* Stream file into cold_data region */
static int stream_file_to_cold(const char *file_path, uint8_t *cold_data, uint64_t *cursor, uint64_t cold_size) {
    int fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open file '%s': %s\n", file_path, strerror(errno));
        return -1;
    }
    
    /* Read in chunks - use smaller buffer to avoid stack overflow */
    uint8_t *buffer = malloc(64 * 1024);  /* 64 KB buffer on heap */
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate buffer\n");
        close(fd);
        return -1;
    }
    ssize_t bytes_read;
    
    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
        if (*cursor + (uint64_t)bytes_read > cold_size) {
            fprintf(stderr, "Error: Cold data overflow\n");
            close(fd);
            return -1;
        }
        
        memcpy(cold_data + *cursor, buffer, (size_t)bytes_read);
        *cursor += (uint64_t)bytes_read;
    }
    
    if (bytes_read < 0) {
        fprintf(stderr, "Error: Read failed for '%s': %s\n", file_path, strerror(errno));
        free(buffer);
        close(fd);
        return -1;
    }
    
    free(buffer);
    close(fd);
    return 0;
}

/* Print usage */
static void print_usage(const char *prog_name) {
    if (!prog_name) prog_name = "melvin_pack_corpus";
    fprintf(stderr, "Usage: %s -i <input_dir> -o <output.m> [options]\n", prog_name);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i, --input <dir>        Root directory of corpus files (required)\n");
    fprintf(stderr, "  -o, --output <path>       Path to melvin.m output (required)\n");
    fprintf(stderr, "  --hot-nodes <N>          Number of hot nodes (default: %d)\n", DEFAULT_HOT_NODES);
    fprintf(stderr, "  --hot-edges <N>          Number of hot edges (default: %d)\n", DEFAULT_HOT_EDGES);
    fprintf(stderr, "  --hot-blob-bytes <B>     Hot blob size in bytes (default: %d)\n", DEFAULT_HOT_BLOB_BYTES);
    fprintf(stderr, "  -h, --help               Show this help\n");
    fflush(stderr);
}

int main(int argc, char *argv[]) {
    const char *input_dir = NULL;
    const char *output_path = NULL;
    uint64_t hot_nodes = DEFAULT_HOT_NODES;
    uint64_t hot_edges = DEFAULT_HOT_EDGES;
    uint64_t hot_blob_bytes = DEFAULT_HOT_BLOB_BYTES;
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: -i requires a directory path\n");
                print_usage(argv[0]);
                return 1;
            }
            input_dir = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: -o requires an output path\n");
                print_usage(argv[0]);
                return 1;
            }
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--hot-nodes") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --hot-nodes requires a number\n");
                print_usage(argv[0]);
                return 1;
            }
            hot_nodes = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--hot-edges") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --hot-edges requires a number\n");
                print_usage(argv[0]);
                return 1;
            }
            hot_edges = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--hot-blob-bytes") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --hot-blob-bytes requires a number\n");
                print_usage(argv[0]);
                return 1;
            }
            hot_blob_bytes = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    /* Validate required arguments */
    if (!input_dir || !output_path) {
        fprintf(stderr, "Error: -i and -o are required\n");
        print_usage(argv[0]);
        return 1;
    }
    
    /* Pass 1: Walk directory and compute total size */
    fprintf(stderr, "=== Pass 1: Computing corpus size ===\n");
    fprintf(stderr, "Scanning directory: %s\n", input_dir);
    fflush(stderr);
    uint64_t total_corpus_bytes = 0;
    
    if (walk_directory(input_dir, NULL, &total_corpus_bytes) < 0) {
        fprintf(stderr, "Error: Failed to walk directory\n");
        corpus_free();
        return 1;
    }
    
    if (total_corpus_bytes == 0) {
        fprintf(stderr, "Error: No files found in corpus directory '%s'\n", input_dir);
        corpus_free();
        return 1;
    }
    
    printf("\nTotal corpus size: %llu bytes (%.2f MB)\n", 
           (unsigned long long)total_corpus_bytes,
           (double)total_corpus_bytes / (1024.0 * 1024.0));
    printf("Files: %zu\n", corpus_list.count);
    
    /* Create v2 melvin.m file */
    printf("\n=== Creating v2 melvin.m file ===\n");
    printf("Hot nodes: %llu\n", (unsigned long long)hot_nodes);
    printf("Hot edges: %llu\n", (unsigned long long)hot_edges);
    printf("Hot blob: %llu bytes\n", (unsigned long long)hot_blob_bytes);
    printf("Cold data: %llu bytes\n", (unsigned long long)total_corpus_bytes);
    
    if (melvin_create_v2(output_path, hot_nodes, hot_edges, hot_blob_bytes, total_corpus_bytes) < 0) {
        fprintf(stderr, "Error: Failed to create melvin.m file '%s'\n", output_path);
        corpus_free();
        return 1;
    }
    
    /* Open the created file and map it */
    printf("\n=== Pass 2: Streaming corpus into cold_data ===\n");
    int fd = open(output_path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open created file '%s': %s\n", output_path, strerror(errno));
        corpus_free();
        return 1;
    }
    
    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "Error: Cannot stat file '%s': %s\n", output_path, strerror(errno));
        close(fd);
        corpus_free();
        return 1;
    }
    
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "Error: mmap failed: %s\n", strerror(errno));
        close(fd);
        corpus_free();
        return 1;
    }
    
    /* Parse header */
    MelvinHeader *hdr = (MelvinHeader *)map;
    if (memcmp(hdr->magic, MELVIN_MAGIC, 4) != 0) {
        fprintf(stderr, "Error: Invalid magic in created file\n");
        munmap(map, (size_t)st.st_size);
        close(fd);
        corpus_free();
        return 1;
    }
    
    if (hdr->version != MELVIN_VERSION) {
        fprintf(stderr, "Error: Wrong version in created file (expected %d, got %u)\n", 
                MELVIN_VERSION, hdr->version);
        munmap(map, (size_t)st.st_size);
        close(fd);
        corpus_free();
        return 1;
    }
    
    /* Get cold_data pointer */
    uint8_t *cold_data = (uint8_t *)map + hdr->cold_data_offset;
    uint64_t cold_size = hdr->cold_data_size;
    
    if (cold_size != total_corpus_bytes) {
        fprintf(stderr, "Error: Cold data size mismatch (expected %llu, got %llu)\n",
                (unsigned long long)total_corpus_bytes, (unsigned long long)cold_size);
        munmap(map, (size_t)st.st_size);
        close(fd);
        corpus_free();
        return 1;
    }
    
    /* Stream all files into cold_data */
    uint64_t cursor = 0;
    for (size_t i = 0; i < corpus_list.count; i++) {
        /* Path is already full path from walk_directory */
        const char *file_path = corpus_list.files[i].path;
        
        /* Extract just filename for display */
        const char *filename = strrchr(file_path, '/');
        filename = filename ? filename + 1 : file_path;
        
        printf("  [%zu/%zu] %s\n", i + 1, corpus_list.count, filename);
        
        if (stream_file_to_cold(file_path, cold_data, &cursor, cold_size) < 0) {
            munmap(map, (size_t)st.st_size);
            close(fd);
            corpus_free();
            return 1;
        }
    }
    
    if (cursor != cold_size) {
        fprintf(stderr, "Error: Size mismatch after streaming (expected %llu, got %llu)\n",
                (unsigned long long)cold_size, (unsigned long long)cursor);
        munmap(map, (size_t)st.st_size);
        close(fd);
        corpus_free();
        return 1;
    }
    
    /* Sync to disk */
    printf("\n=== Syncing to disk ===\n");
    msync(map, (size_t)st.st_size, MS_SYNC);
    
    /* Cleanup */
    munmap(map, (size_t)st.st_size);
    close(fd);
    corpus_free();
    
    printf("\n✓ Successfully packed corpus into '%s'\n", output_path);
    printf("  Total size: %llu bytes (%.2f MB)\n",
           (unsigned long long)st.st_size,
           (double)st.st_size / (1024.0 * 1024.0));
    
    return 0;
}

