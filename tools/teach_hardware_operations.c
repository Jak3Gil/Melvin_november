/*
 * teach_hardware_operations - Teach brain hardware control by feeding ARM64 code
 * 
 * Like preseed_melvin.c but feeds EXECUTABLE CODE instead of text patterns
 * Brain learns operations by having machine code fed as data!
 * 
 * Usage: teach_hardware_operations brain.m
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Forward declare teaching function from melvin.c */
extern uint32_t melvin_teach_operation(Graph *g, const uint8_t *machine_code, 
                                        size_t code_len, const char *name);

/* ARM64 Hardware Control Operations */

/* GPIO Toggle (LED control) - Simple stub for now */
static const uint8_t gpio_toggle_code[] = {
    /* For now: Simple return stub (would be real GPIO code) */
    0x00, 0x00, 0x80, 0xD2,  /* MOV X0, #0 */
    0xC0, 0x03, 0x5F, 0xD6   /* RET */
};

/* Audio Playback Trigger - Stub */
static const uint8_t audio_play_code[] = {
    /* Would call ALSA API or trigger audio buffer */
    0x00, 0x00, 0x80, 0xD2,  /* MOV X0, #0 */
    0xC0, 0x03, 0x5F, 0xD6   /* RET */
};

/* Servo Control - PWM stub */
static const uint8_t servo_control_code[] = {
    /* Would set PWM duty cycle */
    0x00, 0x00, 0x80, 0xD2,  /* MOV X0, #0 */
    0xC0, 0x03, 0x5F, 0xD6   /* RET */
};

/* Addition (for testing) - REAL ARM64 */
static const uint8_t add_code[] = {
    0x00, 0x00, 0x01, 0x8B,  /* ADD X0, X0, X1 */
    0xC0, 0x03, 0x5F, 0xD6   /* RET */
};

/* Multiplication (for testing) */
static const uint8_t mul_code[] = {
    0x00, 0x7C, 0x01, 0x9B,  /* MUL X0, X0, X1 */
    0xC0, 0x03, 0x5F, 0xD6   /* RET */
};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <brain.m>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Teaches brain hardware operations by feeding ARM64 code.\n");
        fprintf(stderr, "Brain stores code in blob and learns when to execute.\n");
        fprintf(stderr, "\n");
        return 1;
    }
    
    const char *brain_path = argv[1];
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  TEACH HARDWARE OPERATIONS                         ║\n");
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║  Feeding ARM64 code to brain as data               ║\n");
    printf("║  Brain becomes self-contained and teachable!       ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");
    
    /* Open brain */
    printf("Opening brain: %s\n", brain_path);
    Graph *g = melvin_open(brain_path, 10000, 50000, 131072);
    
    if (!g) {
        fprintf(stderr, "❌ Failed to open brain: %s\n", brain_path);
        return 1;
    }
    
    printf("✅ Brain opened: %llu nodes, %llu edges\n\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    
    /* Teach operations by feeding ARM64 code */
    printf("═══════════════════════════════════════════════════\n");
    printf("Teaching Hardware Operations\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    uint32_t taught_count = 0;
    uint32_t node_ids[10];
    
    /* Teach arithmetic (for testing/validation) */
    printf("1. Teaching arithmetic operations...\n");
    node_ids[0] = melvin_teach_operation(g, add_code, sizeof(add_code), "EXEC_ADD");
    if (node_ids[0] != UINT32_MAX) taught_count++;
    
    node_ids[1] = melvin_teach_operation(g, mul_code, sizeof(mul_code), "EXEC_MUL");
    if (node_ids[1] != UINT32_MAX) taught_count++;
    
    /* Teach hardware control */
    printf("\n2. Teaching hardware control operations...\n");
    node_ids[2] = melvin_teach_operation(g, gpio_toggle_code, sizeof(gpio_toggle_code), "GPIO_TOGGLE");
    if (node_ids[2] != UINT32_MAX) taught_count++;
    
    node_ids[3] = melvin_teach_operation(g, audio_play_code, sizeof(audio_play_code), "AUDIO_PLAY");
    if (node_ids[3] != UINT32_MAX) taught_count++;
    
    node_ids[4] = melvin_teach_operation(g, servo_control_code, sizeof(servo_control_code), "SERVO_CONTROL");
    if (node_ids[4] != UINT32_MAX) taught_count++;
    
    printf("\n");
    printf("═══════════════════════════════════════════════════\n");
    printf("Summary\n");
    printf("═══════════════════════════════════════════════════\n\n");
    
    printf("✅ Taught %u operations to brain\n", taught_count);
    printf("   Operations stored in blob as ARM64 code\n");
    printf("   EXEC nodes created and pointing to code\n");
    printf("   Brain is now self-contained!\n\n");
    
    printf("EXEC Nodes Created:\n");
    for (uint32_t i = 0; i < taught_count && i < 10; i++) {
        if (node_ids[i] != UINT32_MAX) {
            printf("  - Node %u: blob offset %llu\n", 
                   node_ids[i], 
                   (unsigned long long)g->nodes[node_ids[i]].payload_offset);
        }
    }
    
    printf("\n");
    printf("Next Steps:\n");
    printf("  1. Run: tools/create_port_patterns %s\n", brain_path);
    printf("  2. Run: tools/bootstrap_hardware_edges %s\n", brain_path);
    printf("  3. Deploy to Jetson and run!\n\n");
    
    /* Close and save */
    melvin_close(g);
    
    printf("✅ Brain saved with taught operations!\n\n");
    
    return 0;
}

