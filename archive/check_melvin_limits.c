#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <errno.h>

int main(int argc, char **argv) {
    const char *brain_file = "melvin.m";
    if (argc > 1) {
        brain_file = argv[1];
    }
    
    printf("\n========================================\n");
    printf("MELVIN LIMITS ANALYSIS\n");
    printf("========================================\n\n");
    
    // Check if file exists
    int fd = open(brain_file, O_RDONLY);
    if (fd < 0) {
        printf("File: %s (does not exist yet)\n", brain_file);
        printf("\n");
    } else {
        struct stat st;
        fstat(fd, &st);
        close(fd);
        
        printf("Current brain file: %s\n", brain_file);
        printf("Current size: %zu bytes (%.2f MB)\n", 
               (size_t)st.st_size, st.st_size / 1024.0 / 1024.0);
        printf("\n");
    }
    
    // Check available disk space
    struct statvfs vfs;
    if (statvfs(".", &vfs) == 0) {
        uint64_t available_bytes = (uint64_t)vfs.f_bavail * vfs.f_frsize;
        uint64_t total_bytes = (uint64_t)vfs.f_blocks * vfs.f_frsize;
        uint64_t used_bytes = total_bytes - (uint64_t)vfs.f_bavail * vfs.f_frsize;
        
        printf("Disk Space Analysis:\n");
        printf("  Total disk: %.2f GB\n", total_bytes / 1024.0 / 1024.0 / 1024.0);
        printf("  Used: %.2f GB\n", used_bytes / 1024.0 / 1024.0 / 1024.0);
        printf("  Available: %.2f GB\n", available_bytes / 1024.0 / 1024.0 / 1024.0);
        printf("\n");
        
        // Calculate theoretical limits
        size_t node_size = sizeof(Node);
        size_t edge_size = sizeof(Edge);
        size_t header_size = sizeof(BrainHeader);
        
        // Reserve 10% for safety
        uint64_t usable_space = available_bytes * 90 / 100;
        
        // 30% for nodes, 70% for edges
        uint64_t node_space = usable_space * 30 / 100;
        uint64_t edge_space = usable_space * 70 / 100;
        
        uint64_t max_nodes = node_space / node_size;
        uint64_t max_edges = edge_space / edge_size;
        
        printf("Theoretical Limits (90%% of available space):\n");
        printf("  Maximum nodes: %llu (%.2f billion)\n", 
               (unsigned long long)max_nodes, max_nodes / 1e9);
        printf("  Maximum edges: %llu (%.2f billion)\n", 
               (unsigned long long)max_edges, max_edges / 1e9);
        printf("  Maximum file size: %.2f GB\n", usable_space / 1024.0 / 1024.0 / 1024.0);
        printf("\n");
    } else {
        perror("statvfs");
    }
    
    printf("Code Analysis:\n");
    printf("--------------\n");
    printf("melvin.h BrainHeader:\n");
    printf("  - NO node_cap field\n");
    printf("  - NO edge_cap field\n");
    printf("  - Comment: 'No capacity limits - graph grows organically'\n");
    printf("\n");
    
    printf("melvin.c alloc_node():\n");
    printf("  - NO capacity check\n");
    printf("  - Grows file dynamically when needed\n");
    printf("  - Returns 0 on failure (disk full)\n");
    printf("\n");
    
    printf("melvin.c add_edge():\n");
    printf("  - NO capacity check\n");
    printf("  - Grows file dynamically when needed\n");
    printf("  - Returns silently on failure (disk full)\n");
    printf("\n");
    
    printf("melvin.c grow_graph():\n");
    printf("  - Grows by 50%% or to minimum needed\n");
    printf("  - Fails if ftruncate() fails (disk full)\n");
    printf("  - Returns -1 on failure\n");
    printf("\n");
    
    printf("Actual Limits:\n");
    printf("--------------\n");
    printf("1. Disk space (primary limit)\n");
    printf("   - When disk is full, ftruncate() fails\n");
    printf("   - alloc_node() returns 0\n");
    printf("   - add_edge() returns silently\n");
    printf("\n");
    
    printf("2. Memory-mapped file size (system limit)\n");
    printf("   - OS limits on mmap() size\n");
    printf("   - Typically 2^64 bytes (effectively unlimited)\n");
    printf("\n");
    
    printf("3. uint64_t overflow (theoretical)\n");
    printf("   - num_nodes max: 2^64 - 1 (~18 quintillion)\n");
    printf("   - num_edges max: 2^64 - 1 (~18 quintillion)\n");
    printf("   - Not a practical limit\n");
    printf("\n");
    
    printf("4. Plugin bugs (code issues)\n");
    printf("   - plugins/mc_parse.c checks g->header->node_cap (doesn't exist!)\n");
    printf("   - plugins/mc_parse.c checks g->header->edge_cap (doesn't exist!)\n");
    printf("   - This will always fail (field is 0 or garbage)\n");
    printf("   - BUG: These plugins will stop working\n");
    printf("\n");
    
    printf("========================================\n");
    printf("CONCLUSION\n");
    printf("========================================\n");
    printf("Melvin's ONLY limit is: DISK SPACE\n");
    printf("\n");
    printf("The graph grows organically until:\n");
    printf("  - Disk runs out of space (ftruncate fails)\n");
    printf("  - Or system runs out of memory for mmap\n");
    printf("\n");
    printf("There are NO hardcoded limits in melvin.c\n");
    printf("There are BUGS in some plugins checking non-existent limits\n");
    printf("========================================\n\n");
    
    return 0;
}

