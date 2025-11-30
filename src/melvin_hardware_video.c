/*
 * melvin_hardware_video.c - Hardware video I/O for Melvin
 * 
 * Connects USB cameras to Melvin's soft structure:
 * - Camera → Port 10 (raw frame bytes)
 * - Display ← Port 110 (display output)
 * 
 * Uses V4L2 (Video4Linux2) on Linux/Jetson.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#ifdef __linux__
#include <linux/videodev2.h>
#endif

/* Video configuration */
#define VIDEO_WIDTH 320
#define VIDEO_HEIGHT 240
#define VIDEO_FPS 10  /* 10 frames per second */
#define VIDEO_PORT_INPUT 10    /* Port for camera input */
#define VIDEO_PORT_OUTPUT 110  /* Port for display output */
#define VIDEO_ACTIVATION_THRESHOLD 0.5f
#define MAX_CAMERAS 2

typedef struct {
    int fd;
    void *buffers[4];
    size_t buffer_lengths[4];
    uint32_t n_buffers;
    uint32_t width;
    uint32_t height;
} CameraDevice;

typedef struct {
    Graph *g;
    bool running;
    pthread_t reader_thread;
    pthread_t writer_thread;
    
    CameraDevice cameras[MAX_CAMERAS];
    int n_cameras;
    
    /* Statistics */
    uint64_t frames_read;
    uint64_t frames_written;
} VideoHardware;

static VideoHardware *video_hw = NULL;

#ifdef __linux__
/* Initialize V4L2 camera */
static int init_v4l2_camera(CameraDevice *cam, const char *device_path) {
    struct v4l2_capability cap;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;
    struct v4l2_buffer buf;
    
    cam->fd = open(device_path, O_RDWR | O_NONBLOCK, 0);
    if (cam->fd < 0) {
        fprintf(stderr, "Cannot open camera %s: %s\n", device_path, strerror(errno));
        return -1;
    }
    
    /* Query capabilities */
    if (ioctl(cam->fd, VIDIOC_QUERYCAP, &cap) < 0) {
        fprintf(stderr, "VIDIOC_QUERYCAP failed: %s\n", strerror(errno));
        close(cam->fd);
        return -1;
    }
    
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "Device %s is not a video capture device\n", device_path);
        close(cam->fd);
        return -1;
    }
    
    /* Set format */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = VIDEO_WIDTH;
    fmt.fmt.pix.height = VIDEO_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;  /* MJPEG is common on USB cameras */
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    
    if (ioctl(cam->fd, VIDIOC_S_FMT, &fmt) < 0) {
        fprintf(stderr, "VIDIOC_S_FMT failed: %s\n", strerror(errno));
        close(cam->fd);
        return -1;
    }
    
    cam->width = fmt.fmt.pix.width;
    cam->height = fmt.fmt.pix.height;
    
    /* Request buffers */
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(cam->fd, VIDIOC_REQBUFS, &req) < 0) {
        fprintf(stderr, "VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        close(cam->fd);
        return -1;
    }
    
    if (req.count < 2) {
        fprintf(stderr, "Insufficient buffer memory\n");
        close(cam->fd);
        return -1;
    }
    
    /* Map buffers */
    cam->n_buffers = req.count;
    for (uint32_t i = 0; i < req.count; i++) {
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (ioctl(cam->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            fprintf(stderr, "VIDIOC_QUERYBUF failed: %s\n", strerror(errno));
            close(cam->fd);
            return -1;
        }
        
        cam->buffer_lengths[i] = buf.length;
        cam->buffers[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam->fd, buf.m.offset);
        
        if (cam->buffers[i] == MAP_FAILED) {
            fprintf(stderr, "mmap failed: %s\n", strerror(errno));
            close(cam->fd);
            return -1;
        }
    }
    
    /* Start streaming */
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(cam->fd, VIDIOC_STREAMON, &type) < 0) {
        fprintf(stderr, "VIDIOC_STREAMON failed: %s\n", strerror(errno));
        close(cam->fd);
        return -1;
    }
    
    return 0;
}

/* Read frame from camera */
static int read_camera_frame(CameraDevice *cam, uint8_t *buffer, size_t buffer_size) {
    struct v4l2_buffer buf;
    
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(cam->fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) {
            return 0;  /* No frame available */
        }
        return -1;
    }
    
    size_t frame_size = (size_t)buf.bytesused;
    if (frame_size > buffer_size) {
        frame_size = buffer_size;
    }
    
    memcpy(buffer, cam->buffers[buf.index], frame_size);
    
    if (ioctl(cam->fd, VIDIOC_QBUF, &buf) < 0) {
        return -1;
    }
    
    return (int)frame_size;
}
#endif

/* Video reader thread - reads from cameras and feeds to graph */
static void *video_reader_thread(void *arg) {
    VideoHardware *hw = (VideoHardware *)arg;
    uint8_t *frame_buffer = NULL;
    size_t frame_buffer_size = VIDEO_WIDTH * VIDEO_HEIGHT * 3;  /* RGB */
    
    frame_buffer = malloc(frame_buffer_size);
    if (!frame_buffer) {
        fprintf(stderr, "Failed to allocate frame buffer\n");
        return NULL;
    }
    
    printf("Video reader thread started\n");
    
    uint64_t frame_count = 0;
    uint64_t usec_per_frame = 1000000 / VIDEO_FPS;  /* Microseconds per frame */
    
    while (hw->running) {
        for (int cam_idx = 0; cam_idx < hw->n_cameras; cam_idx++) {
            CameraDevice *cam = &hw->cameras[cam_idx];
            
#ifdef __linux__
            if (cam->fd < 0) continue;
            
            int frame_size = read_camera_frame(cam, frame_buffer, frame_buffer_size);
            if (frame_size > 0) {
                /* Feed frame bytes to graph via port 10 + cam_idx */
                uint32_t port = VIDEO_PORT_INPUT + cam_idx;
                
                /* Feed frame in chunks (don't overwhelm the graph) */
                size_t chunk_size = 256;  /* Feed 256 bytes at a time */
                for (size_t i = 0; i < (size_t)frame_size; i += chunk_size) {
                    size_t chunk = (i + chunk_size < (size_t)frame_size) ? chunk_size : (size_t)frame_size - i;
                    
                    for (size_t j = 0; j < chunk; j++) {
                        float energy = 0.15f;  /* Base energy for video input */
                        melvin_feed_byte(hw->g, port, frame_buffer[i + j], energy);
                    }
                    
                    /* Trigger UEL propagation after each chunk */
                    melvin_call_entry(hw->g);
                }
                
                hw->frames_read++;
                frame_count++;
            }
#else
            /* Fallback: simulate camera frames for non-Linux systems */
            for (size_t i = 0; i < frame_buffer_size; i++) {
                frame_buffer[i] = (uint8_t)(rand() % 256);
            }
            
            uint32_t port = VIDEO_PORT_INPUT + cam_idx;
            for (size_t i = 0; i < frame_buffer_size; i += 256) {
                size_t chunk = (i + 256 < frame_buffer_size) ? 256 : frame_buffer_size - i;
                for (size_t j = 0; j < chunk; j++) {
                    float energy = 0.1f;
                    melvin_feed_byte(hw->g, port, frame_buffer[i + j], energy);
                }
                melvin_call_entry(hw->g);
            }
            hw->frames_read++;
            frame_count++;
#endif
        }
        
        /* Sleep to maintain frame rate */
        usleep(usec_per_frame);
    }
    
    free(frame_buffer);
    printf("Video reader thread stopped\n");
    return NULL;
}

/* Video writer thread - monitors output port and writes to display */
static void *video_writer_thread(void *arg) {
    VideoHardware *hw = (VideoHardware *)arg;
    
    printf("Video writer thread started\n");
    
    while (hw->running) {
        /* Check if output port (110) is activated */
        float activation = melvin_get_activation(hw->g, VIDEO_PORT_OUTPUT);
        
        if (activation > VIDEO_ACTIVATION_THRESHOLD) {
            /* Graph wants to produce video - read from output ports (110-119) */
            /* For now, just acknowledge the output (real display would write to framebuffer) */
            /* TODO: Implement framebuffer/DRM output */
            
            /* Provide positive feedback */
            melvin_feed_byte(hw->g, 30, 1, 0.3f);  /* Positive feedback node */
            hw->frames_written++;
        }
        
        usleep(100000);  /* Check every 100ms */
    }
    
    printf("Video writer thread stopped\n");
    return NULL;
}

/* Initialize video hardware */
int melvin_hardware_video_init(Graph *g, const char **camera_devices, int n_cameras) {
    if (!g) return -1;
    if (n_cameras > MAX_CAMERAS) n_cameras = MAX_CAMERAS;
    
    video_hw = calloc(1, sizeof(VideoHardware));
    if (!video_hw) return -1;
    
    video_hw->g = g;
    video_hw->running = true;
    video_hw->n_cameras = 0;
    
    /* Initialize cameras */
    for (int i = 0; i < n_cameras; i++) {
        if (!camera_devices || !camera_devices[i]) {
            char default_device[32];
            snprintf(default_device, sizeof(default_device), "/dev/video%d", i);
            camera_devices = (const char *[]){default_device};
        }
        
#ifdef __linux__
        if (init_v4l2_camera(&video_hw->cameras[video_hw->n_cameras], camera_devices[i]) == 0) {
            video_hw->n_cameras++;
            printf("Camera %d initialized: %s (%ux%u)\n", 
                   video_hw->n_cameras - 1, camera_devices[i],
                   video_hw->cameras[video_hw->n_cameras - 1].width,
                   video_hw->cameras[video_hw->n_cameras - 1].height);
        } else {
            fprintf(stderr, "Warning: Failed to initialize camera %s\n", camera_devices[i]);
        }
#else
        /* Fallback: simulate camera for non-Linux */
        video_hw->cameras[video_hw->n_cameras].fd = -1;  /* Mark as simulated */
        video_hw->cameras[video_hw->n_cameras].width = VIDEO_WIDTH;
        video_hw->cameras[video_hw->n_cameras].height = VIDEO_HEIGHT;
        video_hw->n_cameras++;
        printf("Camera %d simulated (non-Linux)\n", video_hw->n_cameras - 1);
#endif
    }
    
    if (video_hw->n_cameras == 0) {
        fprintf(stderr, "Warning: No cameras available, using simulation mode\n");
    }
    
    /* Start reader thread (cameras → graph) */
    if (pthread_create(&video_hw->reader_thread, NULL, video_reader_thread, video_hw) != 0) {
        fprintf(stderr, "Failed to create video reader thread\n");
        free(video_hw);
        video_hw = NULL;
        return -1;
    }
    
    /* Start writer thread (graph → display) */
    if (pthread_create(&video_hw->writer_thread, NULL, video_writer_thread, video_hw) != 0) {
        fprintf(stderr, "Failed to create video writer thread\n");
        video_hw->running = false;
        pthread_join(video_hw->reader_thread, NULL);
        free(video_hw);
        video_hw = NULL;
        return -1;
    }
    
    printf("Video hardware initialized\n");
    printf("  Cameras: %d\n", video_hw->n_cameras);
    printf("  Input port: %d-%d\n", VIDEO_PORT_INPUT, VIDEO_PORT_INPUT + video_hw->n_cameras - 1);
    printf("  Output port: %d\n", VIDEO_PORT_OUTPUT);
    
    return 0;
}

/* Shutdown video hardware */
void melvin_hardware_video_shutdown(void) {
    if (!video_hw) return;
    
    video_hw->running = false;
    
    pthread_join(video_hw->reader_thread, NULL);
    pthread_join(video_hw->writer_thread, NULL);
    
#ifdef __linux__
    /* Cleanup cameras */
    for (int i = 0; i < video_hw->n_cameras; i++) {
        CameraDevice *cam = &video_hw->cameras[i];
        if (cam->fd >= 0) {
            enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(cam->fd, VIDIOC_STREAMOFF, &type);
            
            for (uint32_t j = 0; j < cam->n_buffers; j++) {
                if (cam->buffers[j]) {
                    munmap(cam->buffers[j], cam->buffer_lengths[j]);
                }
            }
            close(cam->fd);
        }
    }
#endif
    
    printf("Video hardware shutdown\n");
    printf("  Frames read: %llu\n", (unsigned long long)video_hw->frames_read);
    printf("  Frames written: %llu\n", (unsigned long long)video_hw->frames_written);
    
    free(video_hw);
    video_hw = NULL;
}

/* Get statistics */
void melvin_hardware_video_stats(uint64_t *frames_read, uint64_t *frames_written) {
    if (!video_hw) {
        if (frames_read) *frames_read = 0;
        if (frames_written) *frames_written = 0;
        return;
    }
    
    if (frames_read) *frames_read = video_hw->frames_read;
    if (frames_written) *frames_written = video_hw->frames_written;
}

