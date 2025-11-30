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
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

// Simple DRM/KMS display output for Jetson
// This writes directly to the display via DRM (Direct Rendering Manager)

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

// DRM display state
static int drm_fd = -1;
static uint32_t *framebuffer = NULL;
static uint32_t fb_id = 0;
static uint32_t connector_id = 0;
static uint32_t crtc_id = 0;
static uint32_t width = 1920;
static uint32_t height = 1080;
static int display_initialized = 0;

// Initialize DRM display
static int init_drm_display(void) {
    if (display_initialized) return (framebuffer != NULL) ? 1 : 0;
    
    // Try DRM devices
    const char *drm_devices[] = {"/dev/dri/card0", "/dev/dri/card1", NULL};
    
    for (int i = 0; drm_devices[i]; i++) {
        drm_fd = open(drm_devices[i], O_RDWR);
        if (drm_fd >= 0) {
            // Get resources
            drmModeRes *resources = drmModeGetResources(drm_fd);
            if (resources) {
                // Find connector
                for (int c = 0; c < resources->count_connectors; c++) {
                    drmModeConnector *connector = drmModeGetConnector(drm_fd, resources->connectors[c]);
                    if (connector && connector->connection == DRM_MODE_CONNECTED) {
                        connector_id = connector->connector_id;
                        crtc_id = resources->crtcs[0]; // Use first CRTC
                        
                        // Get mode
                        if (connector->count_modes > 0) {
                            drmModeModeInfo mode = connector->modes[0];
                            width = mode.hdisplay;
                            height = mode.vdisplay;
                        }
                        
                        printf("[mc_display] Found connected display: %dx%d\n", width, height);
                        
                        // Create framebuffer
                        uint32_t handles[4] = {0};
                        uint32_t strides[4] = {0};
                        uint32_t offsets[4] = {0};
                        
                        size_t buffer_size = width * height * 4; // RGBA32
                        // Allocate buffer and create framebuffer
                        // Simplified - would need proper GEM allocation
                        
                        drmModeFreeConnector(connector);
                        drmModeFreeResources(resources);
                        
                        // For now, use simplified approach
                        framebuffer = (uint32_t *)malloc(buffer_size);
                        if (framebuffer) {
                            display_initialized = 1;
                            return 1;
                        }
                    }
                    if (connector) drmModeFreeConnector(connector);
                }
                drmModeFreeResources(resources);
            }
            close(drm_fd);
            drm_fd = -1;
        }
    }
    
    printf("[mc_display] No DRM display found, using console\n");
    display_initialized = 1;
    framebuffer = NULL;
    return 0;
}

// Clear screen (black)
static void clear_screen(void) {
    if (framebuffer) {
        memset(framebuffer, 0, width * height * 4);
    }
}

// Set pixel (RGB)
static void set_pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    if (!framebuffer || x >= width || y >= height) return;
    framebuffer[y * width + x] = (r << 16) | (g << 8) | b;
}

// MC Function: Display graph
void mc_display_graph(Brain *g, uint64_t node_id) {
    static uint64_t last_tick = 0;
    
    if (g->header->tick < last_tick + 10) return;
    last_tick = g->header->tick;
    
    if (!display_initialized) init_drm_display();
    if (!framebuffer) {
        // Console mode
        printf("\033[2J\033[HMelvin Tick: %llu  Nodes: %llu  Edges: %llu\n",
               (unsigned long long)g->header->tick,
               (unsigned long long)g->header->num_nodes,
               (unsigned long long)g->header->num_edges);
        fflush(stdout);
        return;
    }
    
    clear_screen();
    
    // Draw something simple to test
    for (int i = 0; i < 100; i++) {
        set_pixel(i, i, 255, 0, 0); // Red diagonal line
    }
    
    // Would render graph here...
}

void mc_display_init(Brain *g, uint64_t node_id) {
    if (!display_initialized) {
        init_drm_display();
        if (node_id < g->header->node_cap) {
            g->nodes[node_id].a = 1.0f;
            g->nodes[node_id].bias = 5.0f;
        }
    }
}


