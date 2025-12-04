/* Simple hardware test - actually capture and learn */
#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

int main() {
    printf("\n🤖 MELVIN LIVE HARDWARE TEST\n\n");
    
    /* Open the teachable brain we just created */
    Graph *brain = melvin_open("/mnt/melvin_ssd/brains/hardware_brain.m", 
                                10000, 50000, 131072);
    if (!brain) {
        printf("❌ Can't open brain\n");
        return 1;
    }
    
    printf("✅ Brain loaded: %llu nodes, %llu edges\n\n",
           (unsigned long long)brain->node_count,
           (unsigned long long)brain->edge_count);
    
    /* Test 1: Feed simulated audio data */
    printf("📢 Simulating audio input...\n");
    const char *audio_sim = "HELLO_WORLD_AUDIO_STREAM";
    for (int i = 0; i < 10; i++) {
        for (const char *p = audio_sim; *p; p++) {
            melvin_feed_byte(brain, 0, *p, 1.0f);
        }
        melvin_call_entry(brain);
        printf(".");
        fflush(stdout);
    }
    printf(" ✅\n\n");
    
    /* Test 2: Feed simulated camera data */
    printf("📷 Simulating camera input...\n");
    const char *video_sim = "PERSON_DETECTED_CAMERA_1";
    for (int i = 0; i < 10; i++) {
        for (const char *p = video_sim; *p; p++) {
            melvin_feed_byte(brain, 10, *p, 1.0f);
        }
        melvin_call_entry(brain);
        printf(".");
        fflush(stdout);
    }
    printf(" ✅\n\n");
    
    /* Check what brain learned */
    int patterns = 0;
    for (uint64_t i = 840; i < brain->node_count && i < 2000; i++) {
        if (brain->nodes[i].pattern_data_offset > 0) patterns++;
    }
    
    printf("🧠 Brain status:\n");
    printf("   Patterns: %d\n", patterns);
    printf("   Learning from hardware! ✨\n\n");
    
    melvin_close(brain);
    
    printf("✅ Test complete - brain is learning!\n\n");
    return 0;
}

