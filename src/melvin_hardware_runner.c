/*
 * melvin_hardware_runner.c - Complete hardware runner for Melvin
 * 
 * Runs Melvin with real hardware: USB mic, speaker, and cameras.
 * Connects hardware I/O to soft structure ports.
 * 
 * Usage: melvin_hardware_runner <brain.m> [audio_capture] [audio_playback] [camera0] [camera1]
 */

#include "melvin.h"
/* Hardware header not needed - using melvin.h directly */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>

static volatile int running = 1;

static void signal_handler(int sig) {
    (void)sig;
    running = 0;
    printf("\nShutting down...\n");
}

static void print_status(Graph *g, int iteration, 
                         uint64_t audio_read, uint64_t audio_written,
                         uint64_t video_read, uint64_t video_written) {
    if (!g || !g->hdr) return;
    
    /* Check if graph is active (processing) */
    bool is_active = (g->avg_chaos > 0.01f || g->avg_activation > 0.01f);
    const char *status = is_active ? "🟢 ACTIVE" : "⚪ IDLE";
    
    printf("[%d] %s | Nodes: %llu | Edges: %llu | Chaos: %.6f | Activation: %.6f\n",
           iteration, status,
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count,
           g->avg_chaos,
           g->avg_activation);
    printf("      Audio: %llu read, %llu written | Video: %llu read, %llu written\n",
           (unsigned long long)audio_read,
           (unsigned long long)audio_written,
           (unsigned long long)video_read,
           (unsigned long long)video_written);
    
    /* Show conversation indicators */
    if (g->node_count > 310) {
        float stt_activation = fabsf(g->nodes[310].a);
        float llm_activation = (g->node_count > 510) ? fabsf(g->nodes[510].a) : 0.0f;
        float tts_activation = (g->node_count > 610) ? fabsf(g->nodes[610].a) : 0.0f;
        
        if (stt_activation > 0.1f || llm_activation > 0.1f || tts_activation > 0.1f) {
            printf("      💬 Conversation: STT=%.3f LLM=%.3f TTS=%.3f\n",
                   stt_activation, llm_activation, tts_activation);
        }
    }
    
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <brain.m> [audio_capture] [audio_playback] [camera0] [camera1]\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  brain.m: Melvin brain file\n");
        fprintf(stderr, "  audio_capture: ALSA capture device (default: 'default')\n");
        fprintf(stderr, "  audio_playback: ALSA playback device (default: 'default')\n");
        fprintf(stderr, "  camera0: V4L2 camera device (default: '/dev/video0')\n");
        fprintf(stderr, "  camera1: V4L2 camera device (default: '/dev/video1', optional)\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "  %s brain.m\n", argv[0]);
        fprintf(stderr, "  %s brain.m hw:0 hw:0 /dev/video0\n", argv[0]);
        return 1;
    }
    
    const char *brain_path = argv[1];
    const char *audio_capture = (argc > 2) ? argv[2] : "default";
    const char *audio_playback = (argc > 3) ? argv[3] : "default";
    
    /* Collect camera devices */
    const char *camera_devices[2] = {NULL, NULL};
    int n_cameras = 0;
    if (argc > 4) {
        camera_devices[0] = argv[4];
        n_cameras = 1;
    } else {
        camera_devices[0] = "/dev/video0";
        n_cameras = 1;
    }
    if (argc > 5) {
        camera_devices[1] = argv[5];
        n_cameras = 2;
    }
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("========================================\n");
    printf("Melvin Hardware Runner\n");
    printf("========================================\n");
    printf("Brain: %s\n", brain_path);
    printf("Audio capture: %s\n", audio_capture);
    printf("Audio playback: %s\n", audio_playback);
    printf("Cameras: %d\n", n_cameras);
    for (int i = 0; i < n_cameras; i++) {
        printf("  Camera %d: %s\n", i, camera_devices[i]);
    }
    printf("Press Ctrl+C to stop\n\n");
    
    /* Open brain */
    Graph *g = melvin_open(brain_path, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", brain_path);
        fprintf(stderr, "Error: %s\n", strerror(errno));
        perror("melvin_open");
        return 1;
    }
    
    /* Initialize syscalls */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    printf("Brain opened: %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    printf("Soft structure initialized:\n");
    printf("  Input ports: 0-99 (audio: 0, video: 10-19)\n");
    printf("  Output ports: 100-199 (audio: 100, video: 110)\n");
    printf("  Memory ports: 200-255\n");
    printf("\n");
    
    /* Initialize hardware */
    printf("Initializing hardware...\n");
    
    if (melvin_hardware_audio_init(g, audio_capture, audio_playback) < 0) {
        fprintf(stderr, "Warning: Audio hardware initialization failed, continuing without audio\n");
    }
    
    if (melvin_hardware_video_init(g, camera_devices, n_cameras) < 0) {
        fprintf(stderr, "Warning: Video hardware initialization failed, continuing without video\n");
    }
    
    printf("Hardware initialized. Starting continuous processing...\n\n");
    
    /* Main loop - hardware threads do the I/O, this just monitors */
    int iteration = 0;
    time_t last_sync = time(NULL);
    time_t last_status = time(NULL);
    const int SYNC_INTERVAL = 60;  /* Sync every 60 seconds */
    const int STATUS_INTERVAL = 10; /* Print status every 10 seconds */
    
    while (running) {
        time_t now = time(NULL);
        
        /* CONTINUOUS PROCESSING: Call melvin_call_entry to run UEL physics */
        /* This ensures the graph processes continuously, even without new hardware input */
        /* Hardware threads also call this when feeding data, but main loop ensures it keeps running */
        melvin_call_entry(g);
        
        /* Print status periodically */
        if (now - last_status >= STATUS_INTERVAL) {
            uint64_t audio_read = 0, audio_written = 0;
            uint64_t video_read = 0, video_written = 0;
            
            melvin_hardware_audio_stats(&audio_read, &audio_written);
            melvin_hardware_video_stats(&video_read, &video_written);
            
            print_status(g, iteration, audio_read, audio_written, video_read, video_written);
            last_status = now;
        }
        
        /* Sync to disk periodically */
        if (now - last_sync >= SYNC_INTERVAL) {
            melvin_sync(g);
            last_sync = now;
            printf("[%d] Synced to disk\n", iteration);
        }
        
        iteration++;
        usleep(100000);  /* 100ms - faster than 1 second for more responsive processing */
    }
    
    printf("\nShutting down hardware...\n");
    melvin_hardware_audio_shutdown();
    melvin_hardware_video_shutdown();
    
    printf("\nFinal sync...\n");
    melvin_sync(g);
    
    printf("Stopped after %d iterations\n", iteration);
    
    uint64_t audio_read = 0, audio_written = 0;
    uint64_t video_read = 0, video_written = 0;
    melvin_hardware_audio_stats(&audio_read, &audio_written);
    melvin_hardware_video_stats(&video_read, &video_written);
    print_status(g, iteration, audio_read, audio_written, video_read, video_written);
    
    melvin_close(g);
    return 0;
}

