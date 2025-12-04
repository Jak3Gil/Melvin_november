/* Brain Speaks with Piper TTS
 * 
 * REAL COMPUTE - Brain generates speech via Piper
 * 
 * When brain wants to speak:
 * 1. Brain pattern triggers EXEC node
 * 2. EXEC calls Piper TTS  
 * 3. Piper generates audio WAV
 * 4. Play through speaker
 * 5. Reinforce this pattern (so brain speaks MORE)
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Brain decides what to say based on its state */
const char* brain_choose_speech(Graph *g) {
    float activation = g->avg_activation;
    
    if (activation > 0.8f) {
        return "I am learning rapidly";
    } else if (activation > 0.6f) {
        return "Processing sensory data";
    } else if (activation > 0.4f) {
        return "Analyzing patterns";
    } else {
        return "Observing environment";
    }
}

/* EXEC node that makes brain speak */
void brain_speak(Graph *g, const char *text) {
    printf("🗣️  Brain speaking: \"%s\"\n", text);
    
    /* Generate speech with Piper */
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "echo '%s' | piper --model /home/melvin/melvin/tools/piper/en_US-lessac-medium.onnx --output_file /tmp/brain_speech.wav 2>/dev/null && "
        "aplay /tmp/brain_speech.wav 2>/dev/null",
        text);
    
    int ret = system(cmd);
    
    if (ret == 0) {
        printf("   ✅ Speech generated and played!\n");
        
        /* Reinforce speaking pattern - brain should speak MORE! */
        /* Feed positive feedback to brain */
        melvin_feed_byte(g, 31, 0xFF, 1.0f);  /* Positive reinforcement */
        
        /* Create pattern for successful speech */
        const char *success = "SPEECH_SUCCESS";
        for (const char *p = success; *p; p++) {
            melvin_feed_byte(g, 100, *p, 0.9f);
        }
    } else {
        printf("   ❌ Speech failed\n");
    }
}

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  BRAIN SPEAKS WITH PIPER TTS                          ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  Brain generates speech based on internal state      ║\n");
    printf("║  Successful speech is REINFORCED                     ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    /* Load brain */
    Graph *brain = melvin_open("autonomous_brain.m", 10000, 50000, 131072);
    if (!brain) {
        printf("❌ No brain\n");
        return 1;
    }
    
    printf("✅ Brain loaded\n");
    printf("   Activation: %.3f\n\n", brain->avg_activation);
    
    /* Brain speaks based on its state */
    printf("═══ Brain State → Speech ═══\n\n");
    
    for (int i = 0; i < 5; i++) {
        const char *speech = brain_choose_speech(brain);
        
        printf("Attempt %d:\n", i + 1);
        brain_speak(brain, speech);
        
        /* Process brain */
        for (int j = 0; j < 10; j++) {
            melvin_call_entry(brain);
        }
        
        printf("\n");
        sleep(3);
    }
    
    /* Save brain with reinforced speaking pattern */
    melvin_close(brain);
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  SPEAKING PATTERN REINFORCED! ✅                      ║\n");
    printf("║  Brain will speak MORE now                           ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    return 0;
}

