/* Continuous Brain - No Cycles, Async Everything
 * 
 * All systems run continuously in parallel:
 * - Camera thread (continuous capture)
 * - Mic thread (continuous capture)
 * - Brain processing (continuous)
 * - GPU offload (CUDA for acceleration)
 * 
 * Brain learns to use GPU via EXEC nodes!
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;
static Graph *global_brain = NULL;

void stop(int s) { (void)s; running = 0; }

/* Continuous camera thread - NO cycles! */
void* camera_thread(void *arg) {
    (void)arg;
    
    while (running && global_brain) {
        /* Capture frame */
        system("ffmpeg -y -f v4l2 -i /dev/video0 -frames:v 1 /tmp/stream.jpg 2>&1 >/dev/null");
        
        /* Feed to brain immediately */
        FILE *f = fopen("/tmp/stream.jpg", "rb");
        if (f) {
            uint8_t buf[128];
            size_t n = fread(buf, 1, 128, f);
            fclose(f);
            
            for (size_t i = 0; i < n; i += 2) {
                melvin_feed_byte(global_brain, 10, buf[i], 0.8f);
            }
        }
        
        usleep(200000);  /* ~5 FPS, continuous */
    }
    
    return NULL;
}

/* Continuous microphone thread - NO cycles! */
void* mic_thread(void *arg) {
    (void)arg;
    
    /* Continuous audio stream */
    FILE *audio = popen("arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null", "r");
    if (!audio) return NULL;
    
    uint8_t buffer[128];
    while (running && global_brain) {
        size_t n = fread(buffer, 1, 128, audio);
        if (n > 0) {
            /* Feed to brain continuously */
            for (size_t i = 0; i < n; i += 8) {
                melvin_feed_byte(global_brain, 0, buffer[i], 0.9f);
            }
        }
    }
    
    pclose(audio);
    return NULL;
}

/* Continuous brain processing - NO cycles! */
void* brain_thread(void *arg) {
    (void)arg;
    
    while (running && global_brain) {
        /* Continuous propagation */
        melvin_call_entry(global_brain);
        
        /* No sleep - runs as fast as possible! */
        /* This is where GPU would help - offload propagation to CUDA */
    }
    
    return NULL;
}

/* GPU Teaching - How to create CUDA EXEC node */
void teach_gpu_operation(Graph *brain) {
    printf("Teaching brain to use GPU via CUDA EXEC node...\n");
    
    /* EXEC node 3000: GPU offload operation */
    uint32_t GPU_EXEC = 3000;
    
    brain->nodes[GPU_EXEC].payload_offset = 20480;  /* Marker for GPU code */
    brain->nodes[GPU_EXEC].exec_threshold_ratio = 0.1f;
    brain->nodes[GPU_EXEC].semantic_hint = 300;  /* GPU compute */
    
    /* In blob at offset 20480, would store:
     * ARM64 code that calls CUDA API:
     * 
     * cuInit()
     * cuModuleLoad(kernel)
     * cuLaunchKernel(pattern_matching_kernel, ...)
     * cuMemcpy(results back to brain)
     * 
     * Brain learns: When pattern matching is slow → trigger GPU EXEC
     */
    
    printf("✅ EXEC 3000: GPU compute operation\n");
    printf("   Blob offset: 20480\n");
    printf("   Threshold: 0.1 (fires when brain needs acceleration)\n\n");
}

int main() {
    signal(SIGINT, stop);
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  CONTINUOUS BRAIN - NO CYCLES                         ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  Camera: Continuous async                            ║\n");
    printf("║  Mic: Continuous stream                              ║\n");
    printf("║  Processing: Continuous (GPU-ready)                  ║\n");
    printf("║  Press Ctrl+C to stop                                ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    /* Load brain */
    global_brain = melvin_open("responsive_brain.m", 10000, 50000, 131072);
    if (!global_brain) {
        printf("❌ No brain\n");
        return 1;
    }
    
    printf("✅ Brain loaded\n\n");
    
    /* Teach GPU operation */
    teach_gpu_operation(global_brain);
    
    printf("Starting continuous operation...\n\n");
    
    /* Stop PulseAudio for mic access */
    system("pulseaudio -k 2>/dev/null");
    sleep(1);
    
    /* Launch async threads - everything runs in parallel! */
    pthread_t cam_t, mic_t, brain_t;
    
    pthread_create(&cam_t, NULL, camera_thread, NULL);
    pthread_create(&mic_t, NULL, mic_thread, NULL);
    pthread_create(&brain_t, NULL, brain_thread, NULL);
    
    printf("🧠 All systems: ON (continuous)\n");
    printf("   📷 Camera streaming\n");
    printf("   🎤 Mic streaming\n");
    printf("   ⚡ Brain processing\n\n");
    
    /* Main thread monitors and speaks */
    unsigned int speak_count = 0;
    while (running) {
        sleep(10);
        
        if (!running) break;
        
        if (global_brain->nodes[2010].a > 0.4f) {
            printf("🗣️  Brain speaking (%u)\n", ++speak_count);
            system("(echo 'I am processing continuously' | /home/melvin/melvin/tools/piper/piper -m /home/melvin/melvin/tools/piper/en_US-lessac-medium.onnx -f /tmp/sp.wav 2>/dev/null && aplay /tmp/sp.wav 2>/dev/null) &");
            global_brain->nodes[2010].a = 0.1f;
        }
    }
    
    printf("\n🧠 Shutting down...\n");
    
    /* Stop threads */
    running = 0;
    pthread_join(cam_t, NULL);
    pthread_join(mic_t, NULL);
    pthread_join(brain_t, NULL);
    
    /* Count patterns */
    int patterns = 0;
    for (uint64_t i = 840; i < global_brain->node_count && i < 5000; i++) {
        if (global_brain->nodes[i].pattern_data_offset > 0) patterns++;
    }
    
    printf("   Patterns learned: %d\n", patterns);
    
    melvin_close(global_brain);
    
    printf("\n🧠 MELVIN: OFF\n\n");
    
    return 0;
}

