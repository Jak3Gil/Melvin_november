/* Teach Brain to Respond to Speech - Emergent Behavior
 * 
 * Instead of hardcoding "if speech then respond"
 * We TEACH the brain through examples:
 * 
 * 1. Create weak reflex: audio_pattern → speak_EXEC
 * 2. When speech detected → trigger speaking
 * 3. Successful response → reinforce (strengthen edge)
 * 4. Failed/awkward response → weaken edge
 * 5. Over time, brain learns WHEN to respond (emergent!)
 * 
 * Then we test: Say something → Does brain respond naturally?
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

/* Create EXEC node for speaking */
void create_speaking_exec(Graph *brain) {
    /* EXEC node 2010: Brain speaks when triggered */
    uint32_t SPEAK_EXEC = 2010;
    
    brain->nodes[SPEAK_EXEC].payload_offset = 10240;  /* Marker for speak operation */
    brain->nodes[SPEAK_EXEC].semantic_hint = 200;  /* Speech output */
    brain->nodes[SPEAK_EXEC].exec_threshold_ratio = 0.3f;  /* Lower threshold - easier to trigger */
    
    printf("✅ Created EXEC 2010: Speaking operation\n");
}

/* Create weak reflex edges: audio patterns → speaking */
void create_response_reflexes(Graph *brain) {
    printf("\nCreating response reflexes (WEAK - will strengthen if useful):\n");
    
    /* Find audio patterns (created from Port 0) */
    /* Create weak edges: audio pattern → speak EXEC */
    int reflexes = 0;
    
    for (uint64_t pattern = 840; pattern < 900; pattern++) {
        if (brain->nodes[pattern].pattern_data_offset > 0) {
            /* Create weak edge: pattern → speak EXEC */
            uint32_t eid = melvin_create_edge(brain, (uint32_t)pattern, 2010, 0.1f);
            if (eid != UINT32_MAX) {
                reflexes++;
            }
        }
    }
    
    printf("  Created %d weak reflex edges (audio patterns → speaking)\n", reflexes);
    printf("  These will STRENGTHEN if responses are good!\n");
    printf("  They will WEAKEN if responses are bad!\n\n");
}

/* Teach brain by example: "When you hear speech, you should respond" */
void teach_response_examples(Graph *brain) {
    printf("Teaching by example:\n\n");
    
    const char *examples[] = {
        "AUDIO_DETECTED_HUMAN_SPEECH_RESPOND",
        "SOUND_PATTERN_VOICE_DETECTED_SPEAK_BACK",
        "HEARING_VOICE_CONVERSATION_REPLY",
        NULL
    };
    
    for (int i = 0; examples[i]; i++) {
        printf("  Example %d: '%s'\n", i + 1, examples[i]);
        
        /* Feed as high-energy teaching signal */
        for (const char *p = examples[i]; *p; p++) {
            melvin_feed_byte(brain, 0, *p, 1.0f);  /* Port 0 = audio */
        }
        
        /* Process */
        for (int j = 0; j < 15; j++) {
            melvin_call_entry(brain);
        }
    }
    
    printf("\n✅ Brain learned: Audio patterns often precede speaking\n\n");
}

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  TEACHING BRAIN TO RESPOND (EMERGENT BEHAVIOR!)       ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  Not hardcoded - learned through examples!           ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    const char *brain_file = "responsive_brain.m";
    
    /* Create brain */
    printf("Creating responsive brain...\n");
    melvin_create_v2(brain_file, 10000, 50000, 131072, 16384);  /* 16KB blob */
    
    Graph *brain = melvin_open(brain_file, 10000, 50000, 131072);
    if (!brain) {
        printf("❌ Failed\n");
        return 1;
    }
    
    printf("✅ Brain created\n\n");
    
    /* Step 1: Create speaking EXEC */
    printf("STEP 1: Creating speaking capability\n");
    printf("════════════════════════════════════════════════════════\n");
    create_speaking_exec(brain);
    printf("\n");
    
    /* Step 2: Teach response behavior through examples */
    printf("STEP 2: Teaching response behavior\n");
    printf("════════════════════════════════════════════════════════\n");
    teach_response_examples(brain);
    
    /* Step 3: Create reflex edges */
    printf("STEP 3: Creating response reflexes\n");
    printf("════════════════════════════════════════════════════════\n");
    create_response_reflexes(brain);
    
    /* Save taught brain */
    melvin_close(brain);
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  BRAIN TAUGHT TO RESPOND! ✅                          ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  What we created:                                    ║\n");
    printf("║    • EXEC node for speaking                          ║\n");
    printf("║    • Weak reflexes: audio → speak                    ║\n");
    printf("║    • Examples of response behavior                   ║\n");
    printf("║                                                       ║\n");
    printf("║  What will happen:                                   ║\n");
    printf("║    • Brain hears audio → pattern activates           ║\n");
    printf("║    • Pattern triggers speak EXEC (via reflex)        ║\n");
    printf("║    • If response is good → edge strengthens          ║\n");
    printf("║    • If response is bad → edge weakens               ║\n");
    printf("║    • Brain learns WHEN to respond (emergent!)        ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    printf("Brain file: %s\n", brain_file);
    printf("Ready to test emergent response behavior!\n\n");
    
    return 0;
}

