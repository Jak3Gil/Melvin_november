#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <math.h>
#include <time.h>
#include <linux/fb.h>
#include <signal.h>
#include <termios.h>

// Melvin graph structures (from melvin.h)
typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t tick;
    uint64_t node_cap;
    uint64_t edge_cap;
} BrainHeader;

typedef struct {
    float a;
    float bias;
    float decay;
    uint32_t kind;
    uint32_t flags;
    float reliability;
    uint32_t success_count;
    uint32_t failure_count;
    uint32_t mc_id;
    uint16_t mc_flags;
    uint16_t mc_role;
    float value;
} Node;

typedef struct {
    uint64_t src;
    uint64_t dst;
    float w;
    uint32_t flags;
    float elig;
    uint32_t usage_count;
} Edge;

typedef struct {
    BrainHeader *header;
    Node *nodes;
    Edge *edges;
    size_t mmap_size;
    int fd;
} Brain;

// Display state
static int display_fd = -1;
static struct fb_var_screeninfo vinfo;
static struct fb_fix_screeninfo finfo;
static uint32_t *framebuffer = NULL;
static long int screensize = 0;
static int display_initialized = 0;
static uint32_t display_width = 1920;
static uint32_t display_height = 1080;

// Initialize framebuffer display
static int init_framebuffer(void) {
    if (display_initialized) return (framebuffer != NULL) ? 1 : 0;
    
    // Try framebuffer devices in order
    const char *fb_devices[] = {"/dev/fb0", "/dev/fb1", NULL};
    
    for (int i = 0; fb_devices[i]; i++) {
        display_fd = open(fb_devices[i], O_RDWR);
        if (display_fd >= 0) {
            // Get fixed screen info
            if (ioctl(display_fd, FBIOGET_FSCREENINFO, &finfo) == 0) {
                // Get variable screen info
                if (ioctl(display_fd, FBIOGET_VSCREENINFO, &vinfo) == 0) {
                    display_width = vinfo.xres;
                    display_height = vinfo.yres;
                    screensize = finfo.smem_len;
                    
                    printf("[mc_display] Found framebuffer: %s (%dx%d)\n", 
                           fb_devices[i], display_width, display_height);
                    
                    // Map framebuffer to memory
                    framebuffer = (uint32_t *)mmap(0, screensize, 
                                                   PROT_READ | PROT_WRITE, 
                                                   MAP_SHARED, display_fd, 0);
                    if (framebuffer != MAP_FAILED) {
                        printf("[mc_display] Framebuffer mapped successfully\n");
                        display_initialized = 1;
                        return 1;
                    }
                }
            }
            close(display_fd);
            display_fd = -1;
        }
    }
    
    printf("[mc_display] No framebuffer available, using text console\n");
    display_initialized = 1;
    framebuffer = NULL;
    return 0;
}

// Clear screen (black)
static void clear_screen(void) {
    if (framebuffer) {
        memset(framebuffer, 0, screensize);
    }
}

// Draw pixel (RGB)
static void set_pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    if (!framebuffer || x >= display_width || y >= display_height) return;
    
    uint32_t pixel = (r << 16) | (g << 8) | b;
    framebuffer[y * display_width + x] = pixel;
}

// Draw node as circle
static void draw_node(uint32_t x, uint32_t y, float activation, float radius) {
    uint8_t intensity = (uint8_t)(activation * 255.0f);
    uint8_t r = intensity;
    uint8_t g = intensity / 2;
    uint8_t b = intensity;
    
    int r_int = (int)radius;
    for (int dy = -r_int; dy <= r_int; dy++) {
        for (int dx = -r_int; dx <= r_int; dx++) {
            if (dx * dx + dy * dy <= radius * radius) {
                uint32_t px = x + dx;
                uint32_t py = y + dy;
                if (px < display_width && py < display_height) {
                    set_pixel(px, py, r, g, b);
                }
            }
        }
    }
}

// Draw edge as line
static void draw_edge(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, float weight) {
    uint8_t intensity = (uint8_t)(fminf(fabsf(weight) / 10.0f, 1.0f) * 100.0f);
    
    // Simple line drawing (Bresenham)
    int dx = abs((int)x2 - (int)x1);
    int dy = abs((int)y2 - (int)y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    
    int x = x1;
    int y = y1;
    
    while (1) {
        if (x >= 0 && x < display_width && y >= 0 && y < display_height) {
            set_pixel(x, y, intensity / 3, intensity / 3, intensity / 2);
        }
        
        if (x == x2 && y == y2) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

// Simple force-directed layout
static void layout_nodes(Brain *g, float *x_pos, float *y_pos, uint64_t n, int max_iterations) {
    if (n == 0) return;
    
    // Initialize positions randomly
    srand(time(NULL));
    for (uint64_t i = 0; i < n; i++) {
        x_pos[i] = (rand() % display_width);
        y_pos[i] = (rand() % display_height);
    }
    
    // Simple force-directed layout
    float k = sqrtf((display_width * display_height) / (float)n);
    float dt = 0.1f;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        float *fx = (float *)calloc(n, sizeof(float));
        float *fy = (float *)calloc(n, sizeof(float));
        
        if (!fx || !fy) {
            free(fx);
            free(fy);
            return;
        }
        
        // Repulsion between all nodes
        for (uint64_t i = 0; i < n; i++) {
            for (uint64_t j = i + 1; j < n; j++) {
                float dx = x_pos[i] - x_pos[j];
                float dy = y_pos[i] - y_pos[j];
                float d = sqrtf(dx * dx + dy * dy);
                if (d > 0.01f) {
                    float f = (k * k) / d;
                    fx[i] += (dx / d) * f;
                    fy[i] += (dy / d) * f;
                    fx[j] -= (dx / d) * f;
                    fy[j] -= (dy / d) * f;
                }
            }
        }
        
        // Attraction along edges
        uint64_t e_count = g->header->num_edges;
        uint64_t max_edges = (e_count < 5000) ? e_count : 5000;
        for (uint64_t i = 0; i < max_edges; i++) {
            Edge *e = &g->edges[i];
            if (e->src < n && e->dst < n) {
                float dx = x_pos[e->dst] - x_pos[e->src];
                float dy = y_pos[e->dst] - y_pos[e->src];
                float d = sqrtf(dx * dx + dy * dy);
                if (d > 0.01f) {
                    float f = (d * d) / k;
                    fx[e->src] += (dx / d) * f * e->w * 0.1f;
                    fy[e->src] += (dy / d) * f * e->w * 0.1f;
                    fx[e->dst] -= (dx / d) * f * e->w * 0.1f;
                    fy[e->dst] -= (dy / d) * f * e->w * 0.1f;
                }
            }
        }
        
        // Update positions
        for (uint64_t i = 0; i < n; i++) {
            x_pos[i] += fx[i] * dt;
            y_pos[i] += fy[i] * dt;
            
            // Clamp to screen
            x_pos[i] = fmaxf(10.0f, fminf(display_width - 10.0f, x_pos[i]));
            y_pos[i] = fmaxf(10.0f, fminf(display_height - 10.0f, y_pos[i]));
        }
        
        free(fx);
        free(fy);
    }
}

// Render graph to display
static void render_graph(Brain *g) {
    if (!display_initialized) {
        if (!init_framebuffer()) {
            // Console mode - print stats
            printf("\033[2J\033[H"); // Clear screen
            printf("=== Melvin Graph Display ===\n");
            printf("Tick: %llu\n", (unsigned long long)g->header->tick);
            printf("Nodes: %llu/%llu\n", (unsigned long long)g->header->num_nodes, 
                   (unsigned long long)g->header->node_cap);
            printf("Edges: %llu/%llu\n", (unsigned long long)g->header->num_edges,
                   (unsigned long long)g->header->edge_cap);
            fflush(stdout);
            return;
        }
    }
    
    if (!framebuffer) return; // No framebuffer available
    
    clear_screen();
    
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    // Limit display to reasonable number for performance
    uint64_t max_display_nodes = 500;
    uint64_t max_display_edges = 2000;
    
    if (n > max_display_nodes) n = max_display_nodes;
    if (e_count > max_display_edges) e_count = max_display_edges;
    
    // Allocate position arrays
    float *x_pos = (float *)malloc(n * sizeof(float));
    float *y_pos = (float *)malloc(n * sizeof(float));
    
    if (!x_pos || !y_pos) {
        if (x_pos) free(x_pos);
        if (y_pos) free(y_pos);
        return;
    }
    
    // Layout nodes
    layout_nodes(g, x_pos, y_pos, n, 5); // 5 iterations for speed
    
    // Draw edges first (behind nodes)
    for (uint64_t i = 0; i < e_count; i++) {
        Edge *e = &g->edges[i];
        if (e->src < n && e->dst < n && fabsf(e->w) > 0.01f) {
            draw_edge((uint32_t)x_pos[e->src], (uint32_t)y_pos[e->src],
                      (uint32_t)x_pos[e->dst], (uint32_t)y_pos[e->dst],
                      e->w);
        }
    }
    
    // Draw nodes (on top)
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &g->nodes[i];
        if (node->a > 0.05f) { // Only draw active nodes
            float radius = 3.0f + node->a * 8.0f;
            draw_node((uint32_t)x_pos[i], (uint32_t)y_pos[i], node->a, radius);
        }
    }
    
    // Draw title
    char title[256];
    snprintf(title, sizeof(title), "Melvin - Tick: %llu  Nodes: %llu  Edges: %llu",
             (unsigned long long)g->header->tick,
             (unsigned long long)g->header->num_nodes,
             (unsigned long long)g->header->num_edges);
    
    // Draw title text (simple bitmap - just a line for now)
    for (int i = 0; i < 50 && i < display_width; i++) {
        set_pixel(i, 10, 255, 255, 255);
    }
    
    free(x_pos);
    free(y_pos);
}

// MC Function: Display graph on Jetson display port
void mc_display_graph(Brain *g, uint64_t node_id) {
    static uint64_t last_tick = 0;
    
    // Only update display every N ticks for performance
    if (g->header->tick < last_tick + 10) {
        return;
    }
    
    last_tick = g->header->tick;
    
    // Render graph
    render_graph(g);
}

// MC Function: Initialize display
void mc_display_init(Brain *g, uint64_t node_id) {
    if (!display_initialized) {
        init_framebuffer();
        printf("[mc_display] Display system initialized\n");
        
        // Set node activation to trigger continuous display
        if (node_id < g->header->node_cap) {
            g->nodes[node_id].a = 1.0f;
            g->nodes[node_id].bias = 5.0f; // High bias to keep active
        }
    }
}


