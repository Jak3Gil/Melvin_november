/*
 * Unified Architecture Soak Test
 * 
 * Exercises the entire Melvin architecture end-to-end:
 * - Physics (energy, leak, inhibition, homeostasis)
 * - Patterns (creation, matching, explain-away)
 * - EXEC nodes (real_exec_bridge + blob execution)
 * - Prediction framework (internal, sensory, value-delta)
 * - Value scoring (error_delta, compression, control, reward, energy_cost)
 * 
 * Logs comprehensive metrics to CSV files for analysis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "src/melvin.h"

#ifdef __linux__
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <errno.h>
#endif

/* Configuration */
#define DEFAULT_STEPS 1000
#define DEFAULT_SECONDS 60
#define LOG_INTERVAL 10
#define NODE_SAMPLE_INTERVAL 50

/* Camera structure (from test_real_world.c) */
typedef struct {
    int fd;
    uint32_t width;
    uint32_t height;
    uint8_t *buffer;
    size_t buffer_size;
    bool initialized;
} Camera;

/* Test configuration */
typedef struct {
    int max_steps;
    int max_seconds;
    bool use_camera;
    const char *camera_device;
    const char *brain_file;
    bool verbose;
} TestConfig;

/* Representative nodes to sample */
static uint32_t sample_data_nodes[10] = {0};
static uint32_t sample_pattern_nodes[10] = {0};
static uint32_t sample_exec_nodes[10] = {0};
static int num_sample_data = 0;
static int num_sample_pattern = 0;
static int num_sample_exec = 0;

/* Forward declarations */
static bool camera_init(Camera *cam, const char *device);
static bool camera_capture(Camera *cam, uint8_t **out_data, size_t *out_size);
static void camera_close(Camera *cam);
static void generate_synthetic_input(uint8_t *buffer, size_t size, int step);
static void log_global_metrics(FILE *f, Graph *g, int step);
static void log_node_samples(FILE *f, Graph *g, int step);
static void select_representative_nodes(Graph *g);
static void parse_args(int argc, char **argv, TestConfig *config);
static void init_exec_nodes(Graph *g);
extern void init_real_exec_functions(void);

/* Camera functions (simplified from test_real_world.c) */
static bool camera_init(Camera *cam, const char *device) {
#ifdef __linux__
    cam->fd = open(device, O_RDWR | O_NONBLOCK);
    if (cam->fd < 0) {
        return false;
    }
    
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 320;
    fmt.fmt.pix.height = 240;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    
    if (ioctl(cam->fd, VIDIOC_S_FMT, &fmt) < 0) {
        close(cam->fd);
        return false;
    }
    
    cam->width = fmt.fmt.pix.width;
    cam->height = fmt.fmt.pix.height;
    cam->buffer_size = cam->width * cam->height * 2;
    cam->buffer = malloc(cam->buffer_size);
    
    if (!cam->buffer) {
        close(cam->fd);
        return false;
    }
    
    cam->initialized = true;
    return true;
#else
    (void)cam;
    (void)device;
    return false;
#endif
}

static bool camera_capture(Camera *cam, uint8_t **out_data, size_t *out_size) {
#ifdef __linux__
    if (!cam->initialized) return false;
    
    ssize_t n = read(cam->fd, cam->buffer, cam->buffer_size);
    if (n < 0) {
        if (errno == EAGAIN) return false;
        return false;
    }
    
    *out_data = cam->buffer;
    *out_size = (size_t)n;
    return true;
#else
    (void)cam;
    (void)out_data;
    (void)out_size;
    return false;
#endif
}

static void camera_close(Camera *cam) {
    if (cam->initialized) {
#ifdef __linux__
        close(cam->fd);
#endif
        free(cam->buffer);
        cam->initialized = false;
    }
}

/* Generate synthetic structured input */
static void generate_synthetic_input(uint8_t *buffer, size_t size, int step) {
    /* Create structured patterns with occasional anomalies */
    int pattern_cycle = step % 100;
    
    if (pattern_cycle < 25) {
        /* Pattern A: "AAAABBBB" */
        for (size_t i = 0; i < size; i++) {
            buffer[i] = (i % 8 < 4) ? 'A' : 'B';
        }
    } else if (pattern_cycle < 50) {
        /* Pattern B: "CCCCDDDD" */
        for (size_t i = 0; i < size; i++) {
            buffer[i] = (i % 8 < 4) ? 'C' : 'D';
        }
    } else if (pattern_cycle < 75) {
        /* Pattern C: "EEEEFFFF" */
        for (size_t i = 0; i < size; i++) {
            buffer[i] = (i % 8 < 4) ? 'E' : 'F';
        }
    } else {
        /* Anomaly: random noise */
        for (size_t i = 0; i < size; i++) {
            buffer[i] = (uint8_t)((step * 7 + i * 13) % 256);
        }
    }
}

/* Select representative nodes for sampling */
static void select_representative_nodes(Graph *g) {
    num_sample_data = 0;
    num_sample_pattern = 0;
    num_sample_exec = 0;
    
    /* Sample DATA nodes (first 256) */
    for (uint32_t i = 0; i < 256 && i < g->node_count && num_sample_data < 10; i++) {
        if (g->nodes[i].type == NODE_TYPE_DATA) {
            sample_data_nodes[num_sample_data++] = i;
        }
    }
    
    /* Sample PATTERN nodes */
    uint32_t pattern_count = 0;
    for (uint64_t i = 0; i < g->node_count && num_sample_pattern < 10; i++) {
        if (g->nodes[i].type == NODE_TYPE_PATTERN) {
            sample_pattern_nodes[num_sample_pattern++] = (uint32_t)i;
            pattern_count++;
            if (pattern_count >= 20) break;  /* Sample first 20 patterns */
        }
    }
    
    /* Sample EXEC nodes */
    uint32_t exec_count = 0;
    for (uint64_t i = 0; i < g->node_count && num_sample_exec < 10; i++) {
        if (g->nodes[i].type == NODE_TYPE_EXEC) {
            sample_exec_nodes[num_sample_exec++] = (uint32_t)i;
            exec_count++;
            if (exec_count >= 10) break;
        }
    }
    
    printf("Selected %d DATA, %d PATTERN, %d EXEC nodes for sampling\n",
           num_sample_data, num_sample_pattern, num_sample_exec);
}

/* Initialize EXEC nodes in graph */
static void init_exec_nodes(Graph *g) {
    printf("Initializing EXEC nodes...\n");
    
    /* Create EXEC nodes with registered code_ids */
    uint32_t exec_ids[] = {3000, 3001, 3002};  /* Identity, Edge, Blur */
    const char *exec_names[] = {"CPU Identity", "CPU Edge", "GPU Blur"};
    
    for (int i = 0; i < 3; i++) {
        uint32_t exec_id = exec_ids[i];
        if (exec_id >= g->node_count) {
            printf("  Warning: EXEC node %u beyond graph size\n", exec_id);
            continue;
        }
        
        Node *e = &g->nodes[exec_id];
        e->type = NODE_TYPE_EXEC;
        e->code_id = exec_id;
        e->exec_origin = EXEC_ORIGIN_TAUGHT;
        e->created_update = g->physics_step_count;
        e->exec_threshold_ratio = 0.3f;
        e->payload_offset = g->hdr->blob_offset + 1024 * (exec_id % 100);
        
        printf("  Created EXEC node %u: %s (code_id=%u)\n", exec_id, exec_names[i], exec_id);
    }
}

/* Log global metrics to CSV */
static void log_global_metrics(FILE *f, Graph *g, int step) {
    /* Count patterns and EXEC nodes */
    uint32_t total_patterns = 0;
    uint32_t active_patterns = 0;
    float sum_pattern_value = 0.0f;
    float max_pattern_value = 0.0f;
    
    uint32_t total_exec = 0;
    uint32_t exec_fired_this_step = 0;
    float sum_exec_control = 0.0f;
    float sum_exec_value_error = 0.0f;
    static uint32_t prev_exec_counts[1000] = {0};
    
    /* Count prediction errors */
    float sum_pred_error = 0.0f;
    float sum_sensory_error = 0.0f;
    uint32_t pred_error_count = 0;
    uint32_t sensory_error_count = 0;
    
    for (uint64_t i = 0; i < g->node_count; i++) {
        Node *n = &g->nodes[i];
        
        if (n->type == NODE_TYPE_PATTERN) {
            total_patterns++;
            if (n->energy > 0.01f) {
                active_patterns++;
            }
            sum_pattern_value += n->value;
            if (n->value > max_pattern_value) {
                max_pattern_value = n->value;
            }
        }
        
        if (n->type == NODE_TYPE_EXEC) {
            total_exec++;
            if (n->exec_count > prev_exec_counts[i % 1000]) {
                exec_fired_this_step++;
            }
            prev_exec_counts[i % 1000] = n->exec_count;
            sum_exec_control += n->recent_control_value;
            sum_exec_value_error += fabsf(n->value_pred_error);
        }
        
        if (fabsf(n->prediction_error) > 0.001f) {
            sum_pred_error += fabsf(n->prediction_error);
            pred_error_count++;
        }
        
        if (n->sensory_pred_error > 0.001f) {
            sum_sensory_error += n->sensory_pred_error;
            sensory_error_count++;
        }
    }
    
    float mean_pattern_value = (total_patterns > 0) ? (sum_pattern_value / total_patterns) : 0.0f;
    float mean_exec_control = (total_exec > 0) ? (sum_exec_control / total_exec) : 0.0f;
    float mean_exec_value_error = (total_exec > 0) ? (sum_exec_value_error / total_exec) : 0.0f;
    float avg_pred_error = (pred_error_count > 0) ? (sum_pred_error / pred_error_count) : 0.0f;
    float avg_sensory_error = (sensory_error_count > 0) ? (sum_sensory_error / sensory_error_count) : 0.0f;
    
    fprintf(f, "%d,%u,%.4f,%.4f,%.4f,%.4f,%u,%u,%.4f,%.4f,%u,%u,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            step,
            g->active_count,
            g->total_energy,
            g->avg_chaos,
            g->global_value_estimate,
            g->predicted_global_value_delta,
            total_patterns,
            active_patterns,
            mean_pattern_value,
            max_pattern_value,
            total_exec,
            exec_fired_this_step,
            mean_exec_control,
            mean_exec_value_error,
            avg_pred_error,
            avg_sensory_error,
            g->avg_active_count);
    fflush(f);
}

/* Log node samples to CSV */
static void log_node_samples(FILE *f, Graph *g, int step) {
    for (int i = 0; i < num_sample_data; i++) {
        uint32_t node_id = sample_data_nodes[i];
        if (node_id >= g->node_count) continue;
        Node *n = &g->nodes[node_id];
        fprintf(f, "%d,DATA,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                step, node_id, n->energy, n->value, n->prediction_error,
                n->sensory_pred_error, n->recent_error_delta, n->recent_compression_gain,
                n->recent_control_value);
    }
    
    for (int i = 0; i < num_sample_pattern; i++) {
        uint32_t node_id = sample_pattern_nodes[i];
        if (node_id >= g->node_count) continue;
        Node *n = &g->nodes[node_id];
        fprintf(f, "%d,PATTERN,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                step, node_id, n->energy, n->value, n->prediction_error,
                n->sensory_pred_error, n->recent_error_delta, n->recent_compression_gain,
                n->recent_control_value);
    }
    
    for (int i = 0; i < num_sample_exec; i++) {
        uint32_t node_id = sample_exec_nodes[i];
        if (node_id >= g->node_count) continue;
        Node *n = &g->nodes[node_id];
        fprintf(f, "%d,EXEC,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%u,%.4f\n",
                step, node_id, n->energy, n->value, n->prediction_error,
                n->sensory_pred_error, n->recent_error_delta, n->recent_compression_gain,
                n->recent_control_value, n->exec_count, n->exec_success_rate);
    }
    fflush(f);
}

/* Parse command line arguments */
static void parse_args(int argc, char **argv, TestConfig *config) {
    config->max_steps = DEFAULT_STEPS;
    config->max_seconds = DEFAULT_SECONDS;
    config->use_camera = false;
    config->camera_device = "/dev/video0";
    config->brain_file = "unified_test_brain.m";
    config->verbose = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            config->max_steps = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
            config->max_seconds = atoi(argv[i + 1]);
            config->max_steps = 0;  /* Use time-based */
            i++;
        } else if (strcmp(argv[i], "--camera") == 0) {
            config->use_camera = true;
        } else if (strcmp(argv[i], "--no-camera") == 0) {
            config->use_camera = false;
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            config->camera_device = argv[i + 1];
            config->use_camera = true;
            i++;
        } else if (strcmp(argv[i], "--brain") == 0 && i + 1 < argc) {
            config->brain_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            config->verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --steps N        Run for N steps (default: %d)\n", DEFAULT_STEPS);
            printf("  --seconds T      Run for T seconds (default: %d)\n", DEFAULT_SECONDS);
            printf("  --camera         Use USB camera input\n");
            printf("  --no-camera      Use synthetic input (default)\n");
            printf("  --device DEV     Camera device (default: /dev/video0)\n");
            printf("  --brain FILE     Brain file (default: unified_test_brain.m)\n");
            printf("  --verbose, -v    Verbose output\n");
            printf("  --help, -h       Show this help\n");
            exit(0);
        }
    }
}

int main(int argc, char **argv) {
    TestConfig config;
    parse_args(argc, argv, &config);
    
    printf("=== Unified Architecture Soak Test ===\n");
    printf("Brain file: %s\n", config.brain_file);
    printf("Duration: ");
    if (config.max_steps > 0) {
        printf("%d steps\n", config.max_steps);
    } else {
        printf("%d seconds\n", config.max_seconds);
    }
    printf("Input: %s\n", config.use_camera ? "USB Camera" : "Synthetic");
    printf("\n");
    
    /* Initialize real EXEC bridge */
    init_real_exec_functions();
    
    /* Open brain */
    Graph *g = melvin_open(config.brain_file, 50000, 200000, 10*1024*1024);
    if (!g) {
        fprintf(stderr, "Failed to open brain\n");
        return 1;
    }
    
    printf("Graph opened: %llu nodes, %llu edges\n",
           (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    
    /* Initialize EXEC nodes */
    init_exec_nodes(g);
    
    /* Initialize camera if requested */
    Camera cam = {0};
    bool camera_ok = false;
    if (config.use_camera) {
        camera_ok = camera_init(&cam, config.camera_device);
        if (camera_ok) {
            printf("Camera initialized: %dx%d\n", cam.width, cam.height);
        } else {
            printf("Warning: Camera not available, falling back to synthetic input\n");
        }
    }
    
    /* Select representative nodes */
    select_representative_nodes(g);
    
    /* Open log files */
    FILE *metrics_file = fopen("unified_metrics.csv", "w");
    FILE *samples_file = fopen("unified_node_samples.csv", "w");
    
    if (!metrics_file || !samples_file) {
        fprintf(stderr, "Failed to open log files\n");
        melvin_close(g);
        return 1;
    }
    
    /* Write CSV headers */
    fprintf(metrics_file, "step,active_count,total_energy,avg_chaos,global_value,predicted_value_delta,"
            "num_patterns,num_active_patterns,mean_pattern_value,max_pattern_value,"
            "num_exec,exec_fires,mean_exec_control,mean_exec_value_error,"
            "avg_pred_error,avg_sensory_error,avg_active_count\n");
    
    fprintf(samples_file, "step,type,node_id,energy,value,prediction_error,sensory_pred_error,"
            "recent_error_delta,recent_compression_gain,recent_control_value,exec_count,exec_success_rate\n");
    
    /* Main loop */
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    int step = 0;
    uint8_t input_buffer[1000];
    size_t input_size = 100;
    
    printf("\nStarting unified test...\n");
    printf("Logging every %d steps, sampling nodes every %d steps\n", LOG_INTERVAL, NODE_SAMPLE_INTERVAL);
    printf("\n");
    
    while (1) {
        /* Check time limit */
        if (config.max_seconds > 0) {
            gettimeofday(&current_time, NULL);
            double elapsed = (current_time.tv_sec - start_time.tv_sec) +
                           (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
            if (elapsed >= config.max_seconds) {
                break;
            }
        }
        
        /* Check step limit */
        if (config.max_steps > 0 && step >= config.max_steps) {
            break;
        }
        
        /* Get input */
        bool got_input = false;
        if (camera_ok) {
            uint8_t *frame_data = NULL;
            size_t frame_size = 0;
            if (camera_capture(&cam, &frame_data, &frame_size)) {
                /* Downsample frame to manageable size */
                size_t bytes_to_feed = (frame_size > input_size) ? input_size : frame_size;
                for (size_t i = 0; i < bytes_to_feed; i++) {
                    melvin_feed_byte(g, 0, frame_data[i], 0.1f);
                }
                got_input = true;
            }
        }
        
        if (!got_input) {
            /* Generate synthetic input */
            generate_synthetic_input(input_buffer, input_size, step);
            for (size_t i = 0; i < input_size; i++) {
                melvin_feed_byte(g, 0, input_buffer[i], 0.1f);
            }
        }
        
        /* Run physics (includes prediction, value scoring) */
        melvin_run_physics(g);
        
        /* Log metrics */
        if (step % LOG_INTERVAL == 0) {
            log_global_metrics(metrics_file, g, step);
            
            if (config.verbose) {
                /* Count patterns and exec for verbose output */
                uint32_t pattern_count = 0;
                uint32_t exec_count = 0;
                for (uint64_t i = 0; i < g->node_count; i++) {
                    if (g->nodes[i].type == NODE_TYPE_PATTERN && g->nodes[i].energy > 0.01f) {
                        pattern_count++;
                    }
                    if (g->nodes[i].type == NODE_TYPE_EXEC) {
                        exec_count++;
                    }
                }
                printf("Step %d: active=%u, energy=%.3f, chaos=%.3f, value=%.3f, patterns=%u, exec=%u\n",
                       step, g->active_count, g->total_energy, g->avg_chaos,
                       g->global_value_estimate, pattern_count, exec_count);
            }
        }
        
        /* Sample nodes */
        if (step % NODE_SAMPLE_INTERVAL == 0) {
            log_node_samples(samples_file, g, step);
        }
        
        step++;
        
        /* Small delay to prevent overwhelming the system */
        usleep(1000);  /* 1ms */
    }
    
    printf("\n=== Test Complete ===\n");
    printf("Total steps: %d\n", step);
    printf("Final active nodes: %u\n", g->active_count);
    printf("Final node count: %llu\n", (unsigned long long)g->node_count);
    printf("Final edge count: %llu\n", (unsigned long long)g->edge_count);
    printf("\nMetrics written to:\n");
    printf("  - unified_metrics.csv\n");
    printf("  - unified_node_samples.csv\n");
    
    fclose(metrics_file);
    fclose(samples_file);
    camera_close(&cam);
    melvin_close(g);
    
    return 0;
}

