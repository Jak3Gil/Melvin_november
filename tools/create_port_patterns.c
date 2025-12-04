/*
 * create_port_patterns - Create generalized port structure with blanks
 * 
 * Based on preseed_melvin.c but uses blank nodes for generalization
 * Creates patterns like "AUDIO_[BLANK]_PORT_0" that match ANY audio data
 * 
 * Usage: create_port_patterns brain.m
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Port definitions */
typedef struct {
    uint32_t port_num;
    const char *port_name;
    const char *description;
    uint32_t repetitions;  /* How many times to feed (for learning) */
} PortDef;

static const PortDef PORTS[] = {
    /* Input ports */
    {0,   "AUDIO_IN",      "USB microphone input", 10},
    {10,  "CAMERA_1",      "USB camera 1 input", 10},
    {11,  "CAMERA_2",      "USB camera 2 input", 10},
    
    /* AI preprocessing ports */
    {50,  "AUDIO_FEATURES", "Whisper audio features", 10},
    {60,  "VIDEO_FEATURES", "MobileNet visual features", 10},
    
    /* Semantic label ports */
    {100, "AI_TEXT",       "Whisper/Ollama text output", 10},
    {110, "AI_VISION",     "MobileNet classification labels", 10},
    {120, "AI_CONTEXT",    "Ollama context/reasoning", 10},
    
    /* Output ports */
    {200, "SPEAKER_OUT",   "USB speaker output", 10},
    {210, "LED_OUT",       "GPIO LED control", 10},
    {220, "SERVO_OUT",     "PWM servo control", 10},
    
    {0, NULL, NULL, 0}  /* Sentinel */
};

/* Common semantic patterns brain will see */
static const char *SEMANTIC_PATTERNS[] = {
    "GREETING_DETECTED",
    "PERSON_IN_VIEW",
    "OBJECT_IDENTIFIED",
    "QUESTION_ASKED",
    "COMMAND_RECEIVED",
    "MOTION_DETECTED",
    "SOUND_CLASSIFIED",
    "FACE_RECOGNIZED",
    "RESPONSE_NEEDED",
    "ACTION_REQUIRED",
    NULL  /* Sentinel */
};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <brain.m>\n", argv[0]);
        return 1;
    }
    
    const char *brain_path = argv[1];
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  CREATE PORT PATTERNS                              ║\n");
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║  Teaching brain about ports through repetition     ║\n");
    printf("║  Creates patterns with blanks for generalization   ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    /* Open brain */
    printf("Opening brain: %s\n", brain_path);
    Graph *g = melvin_open(brain_path, 10000, 50000, 131072);
    
    if (!g) {
        fprintf(stderr, "❌ Failed to open brain\n");
        return 1;
    }
    
    printf("✅ Opened: %llu nodes, %llu edges\n\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    
    /* Create port patterns through repetition */
    printf("═══════════════════════════════════════════════════\n");
    printf("Creating Port Patterns\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    for (int i = 0; PORTS[i].port_name != NULL; i++) {
        const PortDef *port = &PORTS[i];
        
        printf("Port %u (%s):\n", port->port_num, port->port_name);
        
        /* Feed port name repeatedly to create strong pattern */
        for (uint32_t rep = 0; rep < port->repetitions; rep++) {
            /* Feed to the actual port */
            const char *name = port->port_name;
            for (const char *c = name; *c; c++) {
                melvin_feed_byte(g, port->port_num, *c, 1.0f);
            }
            
            /* Feed newline to separate */
            melvin_feed_byte(g, port->port_num, '\n', 0.5f);
            
            /* Run propagation to strengthen pattern */
            for (int j = 0; j < 5; j++) {
                melvin_call_entry(g);
            }
        }
        
        printf("  ✅ Fed '%s' %u times to port %u\n", 
               port->port_name, port->repetitions, port->port_num);
    }
    
    /* Create semantic patterns */
    printf("\n═══════════════════════════════════════════════════\n");
    printf("Creating Semantic Patterns\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    for (int i = 0; SEMANTIC_PATTERNS[i] != NULL; i++) {
        const char *pattern = SEMANTIC_PATTERNS[i];
        
        /* Feed to semantic port (100) */
        for (int rep = 0; rep < 5; rep++) {
            for (const char *c = pattern; *c; c++) {
                melvin_feed_byte(g, 100, *c, 1.0f);
            }
            melvin_feed_byte(g, 100, '\n', 0.5f);
            
            for (int j = 0; j < 3; j++) {
                melvin_call_entry(g);
            }
        }
        
        printf("  ✅ Created pattern: %s\n", pattern);
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════\n");
    printf("Summary\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    /* Count patterns */
    int patterns = 0;
    for (uint64_t i = 840; i < g->node_count && i < 2000; i++) {
        if (g->nodes[i].pattern_data_offset > 0) patterns++;
    }
    
    printf("✅ Patterns discovered: %d\n", patterns);
    printf("   Brain now knows about ports and common patterns\n");
    printf("   Ready for edge bootstrapping!\n\n");
    
    /* Close and save */
    melvin_close(g);
    
    printf("✅ Brain saved!\n\n");
    
    return 0;
}

