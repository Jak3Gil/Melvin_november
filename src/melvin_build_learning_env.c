/*
 * melvin_build_learning_env.c - Build minimal learning environment for Melvin
 * 
 * Creates a corpus directory with basic computer skills:
 * - Simple C examples (hello, file I/O, syscalls, compile, memory)
 * - Pattern descriptions
 * - Concept explanations
 * 
 * Then packs it into melvin.m's cold_data region.
 * 
 * The graph can read from cold_data when its internal drives trigger it.
 * This is self-directed learning - data is available but not force-fed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <stdint.h>

/* Corpus file structure */
typedef struct {
    char *path;
    size_t size;
} CorpusFile;

static CorpusFile *corpus_files = NULL;
static size_t corpus_count = 0;
static size_t corpus_capacity = 0;

/* Add file to corpus list */
static void add_corpus_file(const char *path, size_t size) {
    if (corpus_count >= corpus_capacity) {
        corpus_capacity = (corpus_capacity == 0) ? 16 : corpus_capacity * 2;
        corpus_files = realloc(corpus_files, corpus_capacity * sizeof(CorpusFile));
        if (!corpus_files) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
    }
    
    corpus_files[corpus_count].path = strdup(path);
    corpus_files[corpus_count].size = size;
    corpus_count++;
}

/* Walk directory and collect files */
static void walk_directory(const char *dir_path, const char *base_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        perror("opendir");
        return;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
        
        struct stat st;
        if (lstat(full_path, &st) != 0) {
            continue;
        }
        
        if (S_ISREG(st.st_mode)) {
            /* Regular file - add to corpus */
            add_corpus_file(full_path, (size_t)st.st_size);
            printf("  %s (%zu bytes)\n", full_path, (size_t)st.st_size);
        } else if (S_ISDIR(st.st_mode)) {
            /* Directory - recurse */
            walk_directory(full_path, base_path);
        }
    }
    
    closedir(dir);
}

int main(int argc, char **argv) {
    const char *corpus_dir = (argc > 1) ? argv[1] : "corpus/basic";
    const char *output_file = (argc > 2) ? argv[2] : "melvin.m";
    
    printf("Building learning environment for Melvin...\n");
    printf("Corpus directory: %s\n", corpus_dir);
    printf("Output file: %s\n", output_file);
    
    /* Check if corpus directory exists */
    struct stat st;
    if (stat(corpus_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Error: Corpus directory '%s' does not exist\n", corpus_dir);
        fprintf(stderr, "Creating basic corpus structure...\n");
        
        /* Create basic corpus if it doesn't exist */
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", corpus_dir);
        if (system(cmd) != 0) {
            fprintf(stderr, "Failed to create corpus directory\n");
            return 1;
        }
        
        fprintf(stderr, "Please add files to %s and run again\n", corpus_dir);
        fprintf(stderr, "Or use: ./melvin_pack_corpus --input %s --output %s\n", corpus_dir, output_file);
        return 1;
    }
    
    /* Walk corpus directory */
    printf("\nCollecting corpus files...\n");
    walk_directory(corpus_dir, corpus_dir);
    
    if (corpus_count == 0) {
        fprintf(stderr, "No files found in corpus directory\n");
        return 1;
    }
    
    /* Calculate total size */
    uint64_t total_size = 0;
    for (size_t i = 0; i < corpus_count; i++) {
        total_size += corpus_files[i].size;
    }
    
    printf("\nCorpus summary:\n");
    printf("  Files: %zu\n", corpus_count);
    printf("  Total size: %llu bytes (%.2f MB)\n", 
           (unsigned long long)total_size, total_size / (1024.0 * 1024.0));
    
    /* Use melvin_pack_corpus to pack it */
    printf("\nPacking corpus into %s...\n", output_file);
    
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), 
             "./melvin_pack_corpus --input %s --output %s --hot-nodes 10000 --hot-edges 50000 --hot-blob-bytes 1048576",
             corpus_dir, output_file);
    
    printf("Running: %s\n", cmd);
    int ret = system(cmd);
    
    if (ret != 0) {
        fprintf(stderr, "Failed to pack corpus\n");
        return 1;
    }
    
    printf("\nLearning environment built!\n");
    printf("Melvin can now read from cold_data when its internal drives trigger it.\n");
    printf("The graph will discover how to use melvin_copy_from_cold syscall.\n");
    printf("This is self-directed learning - data is available but not force-fed.\n");
    
    /* Cleanup */
    for (size_t i = 0; i < corpus_count; i++) {
        free(corpus_files[i].path);
    }
    free(corpus_files);
    
    return 0;
}

