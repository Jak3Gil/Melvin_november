#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <time.h>

// Match melvin.h structure
typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t tick;
    uint64_t node_cap;
    uint64_t edge_cap;
    // ... other fields
} BrainHeader;

int main() {
    const char *brain_file = "/home/melvin/melvin_system/melvin.m";
    // Use /dev/tty0 (console) instead of /dev/tty1 (terminal)
    // The OS terminal works because it uses the console (tty0) mapped to framebuffer
    const char *display_dev = "/dev/tty0";
    
    // Open brain file
    int fd = open(brain_file, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open %s\n", brain_file);
        return 1;
    }
    
    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return 1;
    }
    
    if (st.st_size < sizeof(BrainHeader)) {
        close(fd);
        fprintf(stderr, "Error: Brain file too small\n");
        return 1;
    }
    
    // Map brain file
    void *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return 1;
    }
    
    BrainHeader *h = (BrainHeader*)map;
    
    // Open display device - need write access
    // Try opening with direct write
    int tty_fd = open(display_dev, O_WRONLY | O_NOCTTY);
    if (tty_fd < 0) {
        // Fallback to stdout
        fprintf(stderr, "Warning: Cannot open %s, using stdout\n", display_dev);
        tty_fd = STDOUT_FILENO;
    }
    
    FILE *out = fdopen(tty_fd, "w");
    if (!out) {
        out = stdout;
    }
    
    // Clear screen and show graph stats continuously
    while (1) {
        fprintf(out, "\033[2J\033[H"); // Clear screen and move to top
        
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        char time_str[64];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
        
        fprintf(out, "╔════════════════════════════════════════╗\n");
        fprintf(out, "║     MELVIN GRAPH DISPLAY              ║\n");
        fprintf(out, "╚════════════════════════════════════════╝\n\n");
        
        fprintf(out, "Time: %s\n\n", time_str);
        
        fprintf(out, "Tick: %llu\n", (unsigned long long)h->tick);
        fprintf(out, "─────────────────────────────────────────\n");
        fprintf(out, "Nodes: %llu / %llu (%.1f%%)\n",
            (unsigned long long)h->num_nodes,
            (unsigned long long)h->node_cap,
            h->node_cap > 0 ? (100.0 * h->num_nodes / h->node_cap) : 0.0);
        fprintf(out, "Edges: %llu / %llu (%.1f%%)\n",
            (unsigned long long)h->num_edges,
            (unsigned long long)h->edge_cap,
            h->edge_cap > 0 ? (100.0 * h->num_edges / h->edge_cap) : 0.0);
        fprintf(out, "\n");
        
        fprintf(out, "Status: Melvin is thinking...\n");
        fprintf(out, "\n");
        
        fprintf(out, "─────────────────────────────────────────\n");
        fprintf(out, "Display refresh: Every 2 seconds\n");
        
        fflush(out);
        
        // Refresh every 2 seconds
        sleep(2);
    }
    
    // Cleanup (never reached, but good practice)
    munmap(map, st.st_size);
    close(fd);
    if (out != stdout) fclose(out);
    
    return 0;
}

