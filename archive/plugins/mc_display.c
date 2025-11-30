#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <math.h>
#include <time.h>

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
static char *framebuffer = NULL;
static long int screensize = 0;
static int display_initialized = 0;
static uint32_t display_width = 1920;
static uint32_t display_height = 1080;

// Console fallback
static int console_fd = -1;
static int use_console = 0;

// Initialize framebuffer display
static int init_framebuffer(void) {
    if (display_initialized) return (framebuffer != NULL) ? 1 : 0;
    
#if HAVE_FRAMEBUFFER
    // Try Jetson display framebuffer
    const char *fb_device = "/dev/fb0";
    
    display_fd = open(fb_device, O_RDWR);
    if (display_fd < 0) {
        // Fallback to console
        fprintf(stderr, "[mc_display] Warning: Could not open %s, using console output\n", fb_device);
        use_console = 1;
        console_fd = open("/dev/tty1", O_WRONLY);
        if (console_fd < 0) console_fd = 1; // stdout
        display_initialized = 1;
        framebuffer = NULL;
        return 0; // Use console mode
    }
    
    // Get fixed screen info
    if (ioctl(display_fd, FBIOGET_FSCREENINFO, &finfo) < 0) {
        fprintf(stderr, "[mc_display] Error reading fixed screen info\n");
        close(display_fd);
        display_fd = -1;
        use_console = 1;
        console_fd = open("/dev/tty1", O_WRONLY);
        if (console_fd < 0) console_fd = 1;
        display_initialized = 1;
        framebuffer = NULL;
        return 0;
    }
    
    // Get variable screen info
    if (ioctl(display_fd, FBIOGET_VSCREENINFO, &vinfo) < 0) {
        fprintf(stderr, "[mc_display] Error reading variable screen info\n");
        close(display_fd);
        display_fd = -1;
        use_console = 1;
        console_fd = open("/dev/tty1", O_WRONLY);
        if (console_fd < 0) console_fd = 1;
        display_initialized = 1;
        framebuffer = NULL;
        return 0;
    }
    
    display_width = vinfo.xres;
    display_height = vinfo.yres;
    screensize = finfo.smem_len;
    
    // Try to use framebuffer - if it fails, fall back to console
    // Note: 1920x1080 should work fine with proper display
    
    // Map framebuffer to memory
    framebuffer = (char *)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, display_fd, 0);
    if (framebuffer == MAP_FAILED) {
        fprintf(stderr, "[mc_display] Error mapping framebuffer\n");
        close(display_fd);
        display_fd = -1;
        use_console = 1;
        console_fd = open("/dev/tty1", O_WRONLY);
        if (console_fd < 0) console_fd = 1;
        display_initialized = 1;
        framebuffer = NULL;
        return 0;
    }
    
    printf("[mc_display] Framebuffer initialized: %dx%d, %d bpp\n", 
           display_width, display_height, vinfo.bits_per_pixel);
    display_initialized = 1;
    return 1;
#else
    // No framebuffer support, use console
    use_console = 1;
    console_fd = open("/dev/tty1", O_WRONLY);
    if (console_fd < 0) console_fd = 1;
    display_initialized = 1;
    framebuffer = NULL;
    return 0;
#endif
}

// Convert RGB to framebuffer pixel (assumes 32-bit RGBA)
static inline void set_pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    if (!framebuffer || x >= display_width || y >= display_height) return;
    
    if (vinfo.bits_per_pixel == 32) {
        long int location = (x + vinfo.xoffset) * (vinfo.bits_per_pixel / 8) +
                           (y + vinfo.yoffset) * finfo.line_length;
        
        uint32_t pixel = (r << 16) | (g << 8) | b;
        *((uint32_t *)(framebuffer + location)) = pixel;
    }
}

// Clear screen (black)
static void clear_screen(void) {
    if (framebuffer) {
        memset(framebuffer, 0, screensize);
    }
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
                set_pixel(x + dx, y + dy, r, g, b);
            }
        }
    }
}

// Draw edge as line
static void draw_edge(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, float weight) {
    uint8_t intensity = (uint8_t)(fminf(fabsf(weight) / 10.0f, 1.0f) * 255.0f);
    
    // Simple line drawing (Bresenham-like)
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

// Layout nodes using force-directed algorithm (simplified)
static void layout_nodes(Brain *g, float *x_pos, float *y_pos, int max_iterations) {
    uint64_t n = g->header->num_nodes;
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
            return; // Memory allocation failed
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
        for (uint64_t i = 0; i < e_count; i++) {
            Edge *e = &g->edges[i];
            if (e->src < n && e->dst < n) {
                float dx = x_pos[e->dst] - x_pos[e->src];
                float dy = y_pos[e->dst] - y_pos[e->src];
                float d = sqrtf(dx * dx + dy * dy);
                if (d > 0.01f) {
                    float f = (d * d) / k;
                    fx[e->src] += (dx / d) * f * e->w;
                    fy[e->src] += (dy / d) * f * e->w;
                    fx[e->dst] -= (dx / d) * f * e->w;
                    fy[e->dst] -= (dy / d) * f * e->w;
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
        
        // Free force arrays for this iteration
        free(fx);
        free(fy);
    }
}

// Render graph to console (fallback when framebuffer doesn't work)
static void render_graph_console(Brain *g) {
    char buffer[4096];
    int pos = 0;
    
    // Clear screen and move to top
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "\033[2J\033[H");
    
    // Header
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "╔════════════════════════════════════════════════════╗\n"
                    "║        MELVIN GRAPH VISUALIZATION (Console)        ║\n"
                    "╚════════════════════════════════════════════════════╝\n\n");
    
    // Stats
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "Tick: %llu\n",
                    (unsigned long long)g->header->tick);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "Nodes: %llu / %llu  (%.1f%%)\n",
                    (unsigned long long)g->header->num_nodes,
                    (unsigned long long)g->header->node_cap,
                    g->header->node_cap > 0 ? 
                    (double)g->header->num_nodes / (double)g->header->node_cap * 100.0 : 0.0);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "Edges: %llu / %llu  (%.1f%%)\n\n",
                    (unsigned long long)g->header->num_edges,
                    (unsigned long long)g->header->edge_cap,
                    g->header->edge_cap > 0 ?
                    (double)g->header->num_edges / (double)g->header->edge_cap * 100.0 : 0.0);
    
    // Top active nodes
    uint64_t n = g->header->num_nodes;
    uint64_t max_display = (n < 15) ? n : 15;
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "Top Active Nodes:\n");
    
    // Find and sort active nodes
    uint64_t active_nodes[15];
    uint64_t count = 0;
    
    for (uint64_t i = 0; i < n && count < max_display; i++) {
        if (g->nodes[i].a > 0.1f) {
            active_nodes[count++] = i;
        }
    }
    
    // Simple sort by activation
    for (uint64_t i = 0; i < count; i++) {
        for (uint64_t j = i + 1; j < count; j++) {
            if (g->nodes[active_nodes[j]].a > g->nodes[active_nodes[i]].a) {
                uint64_t tmp = active_nodes[i];
                active_nodes[i] = active_nodes[j];
                active_nodes[j] = tmp;
            }
        }
    }
    
    // Display top nodes with bar chart
    for (uint64_t i = 0; i < count && i < 10; i++) {
        Node *node = &g->nodes[active_nodes[i]];
        float bar_length = node->a * 30.0f;
        
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "  Node %4llu: [",
                        (unsigned long long)active_nodes[i]);
        
        for (int j = 0; j < (int)bar_length && j < 30; j++) {
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, "█");
        }
        for (int j = (int)bar_length; j < 30; j++) {
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, "░");
        }
        
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "] %.3f\n", node->a);
    }
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "\n");
    
    // Write to console
    if (console_fd >= 0) {
        write(console_fd, buffer, pos);
        fsync(console_fd);
    } else {
        write(1, buffer, pos);
        fflush(stdout);
    }
}

// Render graph to display
static void render_graph(Brain *g) {
    if (!display_initialized) {
        init_framebuffer();
    }
    
    // If using console mode (framebuffer doesn't work or is 1920x1080)
    if (use_console || !framebuffer) {
        render_graph_console(g);
        return;
    }
    
    if (!framebuffer) return; // No framebuffer available
    
    clear_screen();
    
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    // Limit display to reasonable number for performance
    uint64_t max_display_nodes = 1000;
    uint64_t max_display_edges = 5000;
    
    if (n > max_display_nodes) n = max_display_nodes;
    if (e_count > max_display_edges) e_count = max_display_edges;
    
    // Allocate position arrays
    float *x_pos = (float *)malloc(n * sizeof(float));
    float *y_pos = (float *)malloc(n * sizeof(float));
    
    if (!x_pos || !y_pos) {
        fprintf(stderr, "[mc_display] Memory allocation failed\n");
        if (x_pos) free(x_pos);
        if (y_pos) free(y_pos);
        return;
    }
    
    // Layout nodes
    layout_nodes(g, x_pos, y_pos, 10); // 10 iterations for speed
    
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
            float radius = 3.0f + node->a * 5.0f;
            draw_node((uint32_t)x_pos[i], (uint32_t)y_pos[i], node->a, radius);
        }
    }
    
    free(x_pos);
    free(y_pos);
}

// MC Function: Display graph on Jetson display port
void mc_display_graph(Brain *g, uint64_t node_id) {
    static uint64_t last_tick = 0;
    static int frame_skip = 0;
    
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

// Draw filled rectangle
static void draw_rect_filled(uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint8_t r, uint8_t g, uint8_t b) {
    for (uint32_t py = y; py < y + h && py < display_height; py++) {
        for (uint32_t px = x; px < x + w && px < display_width; px++) {
            set_pixel(px, py, r, g, b);
        }
    }
}

// Draw rectangle outline
static void draw_rect(uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint8_t r, uint8_t g, uint8_t b) {
    // Top and bottom lines
    for (uint32_t px = x; px < x + w && px < display_width; px++) {
        set_pixel(px, y, r, g, b);
        if (y + h - 1 < display_height) {
            set_pixel(px, y + h - 1, r, g, b);
        }
    }
    // Left and right lines
    for (uint32_t py = y; py < y + h && py < display_height; py++) {
        set_pixel(x, py, r, g, b);
        if (x + w - 1 < display_width) {
            set_pixel(x + w - 1, py, r, g, b);
        }
    }
}

// Simple 8x8 bitmap font (ASCII 32-126)
static const uint8_t font_8x8[95][8] = {
    // Space (32)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    // ! (33)
    {0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00},
    // ... (simplified - would include all ASCII chars)
};

// Draw a single character (8x8)
static void draw_char(uint32_t x, uint32_t y, char c, uint8_t r, uint8_t g, uint8_t b) {
    if (c < 32 || c > 126) c = '?';
    int idx = c - 32;
    
    // Simple 8x8 font rendering
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            // Simple pattern for visibility (would use actual font bitmap)
            if ((row == 0 || row == 7) || (col == 0 || col == 7)) {
                set_pixel(x + col, y + row, r, g, b);
            }
        }
    }
}

// Draw text string
static void draw_text(uint32_t x, uint32_t y, const char *text, uint8_t r, uint8_t g, uint8_t b) {
    uint32_t px = x;
    for (const char *p = text; *p && px < display_width - 8; p++) {
        draw_char(px, y, *p, r, g, b);
        px += 9; // 8 pixels + 1 spacing
    }
}

// Get display dimensions (for external use)
void mc_display_get_size(uint32_t *width, uint32_t *height) {
    if (!display_initialized) init_framebuffer();
    if (width) *width = display_width;
    if (height) *height = display_height;
}

// Public API: Clear display
void mc_display_clear(void) {
    if (!display_initialized) init_framebuffer();
    clear_screen();
}

// Public API: Draw pixel
void mc_display_pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    if (!display_initialized) init_framebuffer();
    set_pixel(x, y, r, g, b);
}

// Public API: Draw rectangle
void mc_display_rect(uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint8_t r, uint8_t g, uint8_t b, int filled) {
    if (!display_initialized) init_framebuffer();
    if (filled) {
        draw_rect_filled(x, y, w, h, r, g, b);
    } else {
        draw_rect(x, y, w, h, r, g, b);
    }
}

// Public API: Draw line
void mc_display_line(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint8_t r, uint8_t g, uint8_t b) {
    if (!display_initialized) init_framebuffer();
    draw_edge(x1, y1, x2, y2, 1.0f); // Reuse draw_edge with weight 1.0
    // Fix color (draw_edge uses grayscale, so we'll create a proper line function)
    int dx = abs((int)x2 - (int)x1);
    int dy = abs((int)y2 - (int)y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    int x = x1;
    int y = y1;
    while (1) {
        if (x >= 0 && x < display_width && y >= 0 && y < display_height) {
            set_pixel(x, y, r, g, b);
        }
        if (x == x2 && y == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 < dx) { err += dx; y += sy; }
    }
}

// Public API: Draw text
void mc_display_text(uint32_t x, uint32_t y, const char *text, uint8_t r, uint8_t g, uint8_t b) {
    if (!display_initialized) init_framebuffer();
    draw_text(x, y, text, r, g, b);
}

// Cleanup on exit
void mc_display_cleanup(void) {
    if (framebuffer && framebuffer != MAP_FAILED) {
        munmap(framebuffer, screensize);
        framebuffer = NULL;
    }
    if (display_fd >= 0) {
        close(display_fd);
        display_fd = -1;
    }
}

