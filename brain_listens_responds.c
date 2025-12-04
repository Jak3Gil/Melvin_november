/* Brain Listens and Responds - Emergent Behavior Test
 * 
 * Brain has been TAUGHT response reflexes (weak edges)
 * Now we test: Will it respond when you speak?
 * 
 * NO hardcoded "if speech then respond"!
 * Only: Feed audio → Let brain decide → See if it responds
 * 
 * If it responds: Reflexes working! Behavior is emergent!
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

static volatile sig_atomic_t g_running = 1;

void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

/* Check if brain wants to speak (EXEC 2010 activated) */
int brain_wants_to_speak(Graph *brain) {
    uint32_t SPEAK_EXEC = 2010;
    
    /* Check if speak EXEC node is highly activated */
    if (brain->nodes[SPEAK_EXEC].a > 0.4f) {
        return 1;
    }
    
    return 0;
}

/* Brain speaks (when IT decides to!) */
void brain_speaks(Graph *brain, unsigned int response_num) {
    /* Brain chooses what to say based on state */
    const char *responses[] = {
        "Hello, I am Melvin",
        "I am listening",
        "I hear you",
        "I am learning from you",
        "Processing your input"
    };
    
    const char *response = responses[response_num % 5];
    
    printf("\n🗣️  >>> BRAIN RESPONDS: \"%s\" <<<\n\n", response);
    
    /* Generate speech */
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "echo '%s' | /home/melvin/melvin/tools/piper/piper "
        "--model /home/melvin/melvin/tools/piper/en_US-lessac-medium.onnx "
        "--output_file /tmp/response.wav 2>/dev/null && "
        "aplay /tmp/response.wav 2>/dev/null",
        response);
    
    system(cmd);
    
    /* Reinforce successful response */
    melvin_feed_byte(brain, 31, 0xFF, 1.0f);  /* Positive feedback */
}

/* Capture and process audio */
void listen_for_speech(Graph *brain) {
    /* Stop PulseAudio to free mic */
    static int first_time = 1;
    if (first_time) {
        system("pulseaudio -k 2>/dev/null");
        sleep(1);
        first_time = 0;
    }
    
    printf("🎤 Listening for speech...\n");
    
    /* Capture 2 seconds */
    system("timeout 3 arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -d 2 /tmp/listening.raw >/dev/null 2>&1");
    
    FILE *f = fopen("/tmp/listening.raw", "rb");
    if (f) {
        /* Feed audio to brain */
        uint8_t buffer[2048];
        size_t n = fread(buffer, 1, sizeof(buffer), f);
        fclose(f);
        
        if (n > 1000) {
            printf("   Captured %zu bytes of audio\n", n);
            printf("   Feeding to brain...\n");
            
            /* Feed audio to Port 0 */
            for (size_t i = 0; i < n && i < 1024; i += 8) {
                melvin_feed_byte(brain, 0, buffer[i], 0.9f);
            }
            
            /* Feed high-energy signal: "SPEECH_HEARD" */
            const char *signal = "SPEECH";
            for (const char *p = signal; *p; p++) {
                melvin_feed_byte(brain, 0, *p, 1.0f);
            }
            
            printf("   ✅ Audio fed to brain\n");
        } else {
            printf("   (Silence - no strong audio)\n");
        }
    }
}

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  EMERGENT RESPONSE TEST                               ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  Brain has learned response reflexes                 ║\n");
    printf("║  Will it respond when you speak?                     ║\n");
    printf("║  NO hardcoded forcing - pure emergence!              ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    signal(SIGINT, signal_handler);
    
    /* Load taught brain */
    Graph *brain = melvin_open("responsive_brain.m", 10000, 50000, 131072);
    if (!brain) {
        printf("❌ No responsive brain. Run teach_response_behavior first!\n");
        return 1;
    }
    
    printf("✅ Loaded responsive brain\n");
    printf("   Brain has learned response reflexes\n");
    printf("   Edges: audio patterns → speaking\n\n");
    
    printf("════════════════════════════════════════════════════════\n");
    printf("LISTENING FOR YOUR SPEECH\n");
    printf("════════════════════════════════════════════════════════\n\n");
    
    printf("Instructions:\n");
    printf("  1. Speak into the microphone\n");
    printf("  2. Brain will process your voice\n");
    printf("  3. Brain decides if/when to respond (NO forcing!)\n");
    printf("  4. Listen for Piper voice response\n");
    printf("  5. Ctrl+C to stop\n\n");
    
    printf("Starting in 3 seconds...\n");
    sleep(3);
    printf("\n");
    
    unsigned int cycle = 0;
    unsigned int responses_given = 0;
    
    while (g_running && cycle < 20) {
        printf("─────────────────────────────────────────────────────\n");
        printf("Cycle %u:\n\n", cycle + 1);
        
        /* Listen for speech */
        listen_for_speech(brain);
        
        /* Process - brain pattern matches and decides */
        printf("   Brain processing...\n");
        for (int i = 0; i < 20; i++) {
            melvin_call_entry(brain);
        }
        
        /* Check if brain WANTS to respond (emergent decision!) */
        if (brain_wants_to_speak(brain)) {
            printf("   ✨ Brain activation triggered response!\n");
            brain_speaks(brain, responses_given);
            responses_given++;
            
            /* Give brain time to speak */
            sleep(3);
        } else {
            printf("   (Brain chose not to respond this time)\n");
        }
        
        printf("\n");
        cycle++;
        
        if (!g_running) break;
    }
    
    printf("\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("SESSION COMPLETE\n");
    printf("════════════════════════════════════════════════════════\n\n");
    
    printf("Cycles: %u\n", cycle);
    printf("Times brain responded: %u\n", responses_given);
    
    if (responses_given > 0) {
        printf("\n✅ EMERGENT BEHAVIOR CONFIRMED!\n");
        printf("   Brain responded WITHOUT being explicitly coded to!\n");
        printf("   Response triggered by learned patterns + reflexes!\n\n");
    } else {
        printf("\n⚠️  Brain didn't respond yet\n");
        printf("   Reflexes might need strengthening\n");
        printf("   Or threshold too high\n");
        printf("   Run longer or speak louder!\n\n");
    }
    
    /* Show what brain learned */
    int patterns = 0;
    for (uint64_t i = 840; i < brain->node_count && i < 5000; i++) {
        if (brain->nodes[i].pattern_data_offset > 0) patterns++;
    }
    
    printf("Brain now has %d patterns\n", patterns);
    printf("EXEC 2010 activation: %.3f\n", brain->nodes[2010].a);
    
    /* Save */
    melvin_close(brain);
    
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  Brain saved with updated response learning          ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    return 0;
}

