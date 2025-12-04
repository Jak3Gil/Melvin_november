/* Melvin Hardware Integration - LIVE with Real I/O
 * 
 * Real hardware learning in action:
 * - Microphone input (audio streaming)
 * - Camera input (visual streaming)
 * - Speaker output (audio feedback)
 * - Pattern learning from real sensory data
 * - Reinforcement learning from execution
 * - All 4 learning mechanisms active simultaneously
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

/* Global state */
static Graph *g_brain = NULL;
static volatile sig_atomic_t g_running = 1;

/* Statistics */
static unsigned long g_audio_samples = 0;
static unsigned long g_camera_frames = 0;
static unsigned long g_patterns_created = 0;
static unsigned long g_exec_successes = 0;
static unsigned long g_exec_failures = 0;

/* Signal handler for clean shutdown */
static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
    printf("\n\n🛑 Shutdown signal received...\n");
}

/* Audio capture thread - reads from microphone */
static void* audio_capture_thread(void *arg) {
    (void)arg;
    
    printf("🎤 Audio capture thread started\n");
    
    /* Try to open audio device */
    FILE *audio = popen("arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 2>/dev/null", "r");
    if (!audio) {
        printf("⚠️  Could not open microphone, using simulated audio\n");
        
        /* Simulated audio input */
        unsigned int sim_cycle = 0;
        while (g_running && g_brain) {
            const char *patterns[] = {
                "AMBIENT_SOUND_DETECTED",
                "VOICE_PATTERN_HEARD",
                "QUIET_BACKGROUND_NOISE",
                "MOTION_AUDIO_SIGNATURE",
                "RHYTHMIC_SOUND_PATTERN"
            };
            
            const char *pattern = patterns[sim_cycle % 5];
            for (const char *p = pattern; *p && g_running; p++) {
                melvin_feed_byte(g_brain, 0, *p, 0.8f);  /* Audio port 0 */
                g_audio_samples++;
            }
            
            sim_cycle++;
            usleep(100000);  /* 100ms between patterns */
        }
        return NULL;
    }
    
    /* Real audio capture */
    unsigned char buffer[256];
    while (g_running && g_brain) {
        size_t n = fread(buffer, 1, sizeof(buffer), audio);
        if (n > 0) {
            /* Feed raw audio samples to brain */
            for (size_t i = 0; i < n; i++) {
                melvin_feed_byte(g_brain, 0, buffer[i], 0.9f);  /* Audio port 0 */
                g_audio_samples++;
            }
        } else {
            usleep(10000);  /* 10ms */
        }
    }
    
    pclose(audio);
    printf("🎤 Audio capture stopped\n");
    return NULL;
}

/* Camera capture thread - reads from camera */
static void* camera_capture_thread(void *arg) {
    (void)arg;
    
    printf("📷 Camera capture thread started\n");
    
    /* Try to open camera */
    int camera_fd = open("/dev/video0", O_RDONLY);
    if (camera_fd < 0) {
        printf("⚠️  Could not open camera, using simulated video\n");
        
        /* Simulated camera input */
        unsigned int frame = 0;
        while (g_running && g_brain) {
            const char *patterns[] = {
                "VISUAL_MOTION_DETECTED",
                "STATIC_SCENE_OBSERVED",
                "LIGHT_CHANGE_SENSED",
                "OBJECT_IN_FRAME",
                "EDGE_PATTERN_FOUND"
            };
            
            const char *pattern = patterns[frame % 5];
            for (const char *p = pattern; *p && g_running; p++) {
                melvin_feed_byte(g_brain, 10, *p, 0.7f);  /* Camera port 10 */
                g_camera_frames++;
            }
            
            frame++;
            usleep(33000);  /* ~30 FPS */
        }
        return NULL;
    }
    
    /* Real camera capture */
    unsigned char buffer[1024];
    while (g_running && g_brain) {
        ssize_t n = read(camera_fd, buffer, sizeof(buffer));
        if (n > 0) {
            /* Feed image data to brain - sample every Nth byte */
            for (ssize_t i = 0; i < n; i += 32) {  /* Sample rate */
                melvin_feed_byte(g_brain, 10, buffer[i], 0.8f);  /* Camera port 10 */
                g_camera_frames++;
            }
        } else {
            usleep(10000);  /* 10ms */
        }
    }
    
    close(camera_fd);
    printf("📷 Camera capture stopped\n");
    return NULL;
}

/* Processing thread - runs graph propagation and execution */
static void* processing_thread(void *arg) {
    (void)arg;
    
    printf("🧠 Brain processing thread started\n");
    
    unsigned int cycle = 0;
    unsigned int last_pattern_count = 0;
    
    while (g_running && g_brain) {
        /* Run one propagation cycle */
        melvin_call_entry(g_brain);
        cycle++;
        
        /* Check for new patterns every 10 cycles */
        if (cycle % 10 == 0) {
            unsigned int pattern_count = 0;
            for (uint64_t i = 840; i < g_brain->node_count && i < 5000; i++) {
                if (g_brain->nodes[i].pattern_data_offset > 0) {
                    pattern_count++;
                }
            }
            
            if (pattern_count > last_pattern_count) {
                g_patterns_created = pattern_count;
                last_pattern_count = pattern_count;
            }
        }
        
        /* Check EXEC node statistics */
        if (cycle % 50 == 0) {
            unsigned int successes = 0;
            unsigned int failures = 0;
            
            for (uint64_t i = 2000; i < g_brain->node_count && i < 2100; i++) {
                if (g_brain->nodes[i].payload_offset > 0 && 
                    g_brain->nodes[i].exec_count > 0) {
                    if (g_brain->nodes[i].exec_success_rate > 0.5f) {
                        successes++;
                    } else {
                        failures++;
                    }
                }
            }
            
            g_exec_successes = successes;
            g_exec_failures = failures;
        }
        
        /* Adaptive sleep based on activity */
        usleep(1000);  /* 1ms base cycle time */
    }
    
    printf("🧠 Brain processing stopped\n");
    return NULL;
}

/* Output thread - produces audio feedback */
static void* output_thread(void *arg) {
    (void)arg;
    
    printf("🔊 Audio output thread started\n");
    
    /* Check if we can output audio */
    int has_speaker = (system("aplay -l 2>/dev/null | grep -q 'USB Audio'") == 0);
    
    unsigned int last_patterns = 0;
    unsigned int beep_count = 0;
    
    while (g_running && g_brain) {
        /* Beep when new patterns are learned */
        if (g_patterns_created > last_patterns) {
            unsigned int new_patterns = g_patterns_created - last_patterns;
            printf("🎵 Pattern learned! (total: %lu)\n", g_patterns_created);
            
            if (has_speaker && beep_count < 10) {  /* Limit beeps */
                /* Play a tone to indicate learning */
                system("timeout 0.1 speaker-test -t sine -f 800 -l 1 >/dev/null 2>&1 &");
                beep_count++;
            }
            
            last_patterns = g_patterns_created;
        }
        
        sleep(1);
    }
    
    printf("🔊 Audio output stopped\n");
    return NULL;
}

/* Status display thread */
static void* status_thread(void *arg) {
    (void)arg;
    
    time_t start_time = time(NULL);
    
    while (g_running) {
        sleep(5);  /* Update every 5 seconds */
        
        if (!g_running) break;
        
        time_t elapsed = time(NULL) - start_time;
        unsigned int minutes = elapsed / 60;
        unsigned int seconds = elapsed % 60;
        
        printf("\n");
        printf("╔════════════════════════════════════════════════════╗\n");
        printf("║  MELVIN LIVE - Runtime: %02um %02us                   ║\n", minutes, seconds);
        printf("╠════════════════════════════════════════════════════╣\n");
        printf("║  🎤 Audio samples: %-10lu                   ║\n", g_audio_samples);
        printf("║  📷 Camera frames: %-10lu                   ║\n", g_camera_frames);
        printf("║  📊 Patterns created: %-10lu                ║\n", g_patterns_created);
        printf("║  ✅ EXEC successes: %-10lu                  ║\n", g_exec_successes);
        printf("║  ❌ EXEC failures: %-10lu                   ║\n", g_exec_failures);
        
        if (g_brain) {
            printf("║  🧠 Brain: %llu nodes, %llu edges              ║\n", 
                   (unsigned long long)g_brain->node_count,
                   (unsigned long long)g_brain->edge_count);
        }
        
        printf("╚════════════════════════════════════════════════════╝\n");
        printf("\n");
    }
    
    return NULL;
}

int main(int argc, char **argv) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  MELVIN HARDWARE INTEGRATION - LIVE SYSTEM        ║\n");
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║  Real-time learning from hardware I/O             ║\n");
    printf("║  Multiple learning mechanisms active              ║\n");
    printf("║  Press Ctrl+C to stop                             ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    /* Get brain file from command line or use default */
    const char *brain_file = (argc > 1) ? argv[1] : "hardware_brain.m";
    
    /* Load brain */
    printf("📂 Loading brain: %s\n", brain_file);
    g_brain = melvin_open(brain_file, 10000, 50000, 131072);
    
    if (!g_brain) {
        printf("❌ Could not open brain file\n");
        printf("   Looking for: %s\n", brain_file);
        printf("   Make sure the brain file exists and is accessible.\n\n");
        return 1;
    }
    
    printf("✅ Brain loaded: %llu nodes, %llu edges\n\n",
           (unsigned long long)g_brain->node_count,
           (unsigned long long)g_brain->edge_count);
    
    /* Count initial patterns and EXEC nodes */
    unsigned int initial_patterns = 0;
    unsigned int exec_nodes = 0;
    
    for (uint64_t i = 840; i < g_brain->node_count && i < 5000; i++) {
        if (g_brain->nodes[i].pattern_data_offset > 0) {
            initial_patterns++;
        }
    }
    
    for (uint64_t i = 2000; i < g_brain->node_count && i < 2100; i++) {
        if (g_brain->nodes[i].payload_offset > 0) {
            exec_nodes++;
        }
    }
    
    printf("📊 Initial state:\n");
    printf("   Patterns: %u\n", initial_patterns);
    printf("   EXEC nodes: %u\n", exec_nodes);
    printf("\n");
    
    g_patterns_created = initial_patterns;
    
    /* Set up signal handlers for clean shutdown */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Launch threads */
    printf("🚀 Starting subsystems...\n\n");
    
    pthread_t audio_thread, camera_thread, process_thread, out_thread, stat_thread;
    
    pthread_create(&audio_thread, NULL, audio_capture_thread, NULL);
    usleep(100000);
    
    pthread_create(&camera_thread, NULL, camera_capture_thread, NULL);
    usleep(100000);
    
    pthread_create(&process_thread, NULL, processing_thread, NULL);
    usleep(100000);
    
    pthread_create(&out_thread, NULL, output_thread, NULL);
    usleep(100000);
    
    pthread_create(&stat_thread, NULL, status_thread, NULL);
    
    printf("✅ All subsystems running!\n");
    printf("🧠 Brain is now learning from real hardware...\n\n");
    
    /* Main loop - just wait for shutdown signal */
    while (g_running) {
        sleep(1);
    }
    
    /* Shutdown */
    printf("\n📊 Shutting down gracefully...\n");
    
    /* Wait for threads to finish */
    pthread_join(audio_thread, NULL);
    pthread_join(camera_thread, NULL);
    pthread_join(process_thread, NULL);
    pthread_join(out_thread, NULL);
    pthread_join(stat_thread, NULL);
    
    /* Final statistics */
    printf("\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  FINAL STATISTICS                                  ║\n");
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║  🎤 Total audio samples: %-10lu             ║\n", g_audio_samples);
    printf("║  📷 Total camera frames: %-10lu             ║\n", g_camera_frames);
    printf("║  📊 Total patterns: %-10lu                  ║\n", g_patterns_created);
    printf("║  ✅ EXEC successes: %-10lu                  ║\n", g_exec_successes);
    printf("║  ❌ EXEC failures: %-10lu                   ║\n", g_exec_failures);
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    /* Show some learned patterns */
    printf("🎓 Patterns learned (sample):\n");
    int shown = 0;
    for (uint64_t i = 840; i < g_brain->node_count && i < 2000 && shown < 10; i++) {
        if (g_brain->nodes[i].pattern_data_offset > 0) {
            printf("   Pattern %llu: activation=%.3f\n", 
                   (unsigned long long)i, 
                   g_brain->nodes[i].a);
            shown++;
        }
    }
    printf("\n");
    
    /* Show EXEC node learning */
    printf("🎯 Reinforcement Learning Results:\n");
    for (uint64_t i = 2000; i < g_brain->node_count && i < 2100; i++) {
        if (g_brain->nodes[i].payload_offset > 0 && g_brain->nodes[i].exec_count > 0) {
            printf("   Node %llu: executions=%u, success=%.3f, threshold=%.3f",
                   (unsigned long long)i,
                   g_brain->nodes[i].exec_count,
                   g_brain->nodes[i].exec_success_rate,
                   g_brain->nodes[i].exec_threshold_ratio);
            
            if (g_brain->nodes[i].exec_success_rate > 0.7f) {
                printf(" ✅ REINFORCED\n");
            } else if (g_brain->nodes[i].exec_success_rate < 0.3f) {
                printf(" ❌ SUPPRESSED\n");
            } else {
                printf(" ⚖️  LEARNING\n");
            }
        }
    }
    printf("\n");
    
    /* Close brain */
    melvin_close(g_brain);
    
    printf("✅ Brain saved and closed\n");
    printf("🎉 Melvin hardware session complete!\n\n");
    
    return 0;
}

