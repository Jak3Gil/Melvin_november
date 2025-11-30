/*
 * test_conversation_path.c - Verify conversation path is ready
 * 
 * Checks: Mic (port 0) → STT (300) → LLM (500) → TTS (600) → Speaker (port 100)
 */

#include "melvin.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

static bool has_edge_path(Graph *g, uint32_t from, uint32_t to) {
    for (uint64_t i = 0; i < g->edge_count; i++) {
        if (g->edges[i].src == from && g->edges[i].dst == to) {
            return true;
        }
    }
    return false;
}

static void check_conversation_path(Graph *g) {
    printf("\n========================================\n");
    printf("Conversation Path Check\n");
    printf("========================================\n");
    printf("\nExpected path:\n");
    printf("  Mic (port 0) → STT Gateway (300) → STT Output (310)\n");
    printf("  STT Output (310) → Conversation Memory (204-209)\n");
    printf("  Conversation Memory (204-209) → LLM Gateway (500) → LLM Output (510)\n");
    printf("  LLM Output (510) → Conversation Memory (204-209)\n");
    printf("  Conversation Memory (204-209) → TTS Gateway (600) → TTS Output (610)\n");
    printf("  TTS Output (610) → Speaker (port 100)\n");
    printf("\n");
    
    int missing = 0;
    
    /* Check: Port 0 → STT Gateway (300) */
    bool found = false;
    for (uint32_t stt = 300; stt < 310; stt++) {
        if (has_edge_path(g, 0, stt)) {
            printf("  ✓ Port 0 → STT Gateway (%u)\n", stt);
            found = true;
            break;
        }
    }
    if (!found) {
        printf("  ✗ Port 0 → STT Gateway (300-309) - MISSING\n");
        missing++;
    }
    
    /* Check: STT Gateway → STT Output (310) */
    bool stt_to_output = false;
    for (uint32_t stt_in = 300; stt_in < 310; stt_in++) {
        if (has_edge_path(g, stt_in, 310)) {
            printf("  ✓ STT Gateway (%u) → STT Output (310)\n", stt_in);
            stt_to_output = true;
            break;
        }
    }
    if (!stt_to_output) {
        printf("  ✗ STT Gateway → STT Output (310) - MISSING\n");
        missing++;
    }
    
    /* Check: STT Output (310) → Conversation Memory (204-209) */
    bool stt_to_conv = false;
    for (uint32_t conv = 204; conv < 210; conv++) {
        if (has_edge_path(g, 310, conv)) {
            printf("  ✓ STT Output (310) → Conversation Memory (%u)\n", conv);
            stt_to_conv = true;
            break;
        }
    }
    if (!stt_to_conv) {
        printf("  ✗ STT Output (310) → Conversation Memory (204-209) - MISSING\n");
        missing++;
    }
    
    /* Check: Conversation Memory → LLM Gateway (500) */
    bool conv_to_llm = false;
    for (uint32_t conv = 204; conv < 210; conv++) {
        for (uint32_t llm = 500; llm < 510; llm++) {
            if (has_edge_path(g, conv, llm)) {
                printf("  ✓ Conversation Memory (%u) → LLM Gateway (%u)\n", conv, llm);
                conv_to_llm = true;
                break;
            }
        }
        if (conv_to_llm) break;
    }
    if (!conv_to_llm) {
        printf("  ✗ Conversation Memory → LLM Gateway (500-509) - MISSING\n");
        missing++;
    }
    
    /* Check: LLM Gateway → LLM Output (510) */
    bool llm_to_output = false;
    for (uint32_t llm_in = 500; llm_in < 510; llm_in++) {
        if (has_edge_path(g, llm_in, 510)) {
            printf("  ✓ LLM Gateway (%u) → LLM Output (510)\n", llm_in);
            llm_to_output = true;
            break;
        }
    }
    if (!llm_to_output) {
        printf("  ✗ LLM Gateway → LLM Output (510) - MISSING\n");
        missing++;
    }
    
    /* Check: LLM Output (510) → Conversation Memory */
    bool llm_to_conv = false;
    for (uint32_t conv = 204; conv < 210; conv++) {
        if (has_edge_path(g, 510, conv)) {
            printf("  ✓ LLM Output (510) → Conversation Memory (%u)\n", conv);
            llm_to_conv = true;
            break;
        }
    }
    if (!llm_to_conv) {
        printf("  ✗ LLM Output (510) → Conversation Memory - MISSING\n");
        missing++;
    }
    
    /* Check: Conversation Memory → TTS Gateway (600) */
    bool conv_to_tts = false;
    for (uint32_t conv = 204; conv < 210; conv++) {
        for (uint32_t tts = 600; tts < 610; tts++) {
            if (has_edge_path(g, conv, tts)) {
                printf("  ✓ Conversation Memory (%u) → TTS Gateway (%u)\n", conv, tts);
                conv_to_tts = true;
                break;
            }
        }
        if (conv_to_tts) break;
    }
    if (!conv_to_tts) {
        printf("  ✗ Conversation Memory → TTS Gateway (600-609) - MISSING\n");
        missing++;
    }
    
    /* Check: TTS Gateway → TTS Output (610) */
    bool tts_to_output = false;
    for (uint32_t tts_in = 600; tts_in < 610; tts_in++) {
        if (has_edge_path(g, tts_in, 610)) {
            printf("  ✓ TTS Gateway (%u) → TTS Output (610)\n", tts_in);
            tts_to_output = true;
            break;
        }
    }
    if (!tts_to_output) {
        printf("  ✗ TTS Gateway → TTS Output (610) - MISSING\n");
        missing++;
    }
    
    /* Check: TTS Output (610) → Port 100 (Speaker) */
    if (has_edge_path(g, 610, 100)) {
        printf("  ✓ TTS Output (610) → Speaker (port 100)\n");
    } else {
        printf("  ✗ TTS Output (610) → Speaker (port 100) - MISSING\n");
        missing++;
    }
    
    printf("\n");
    if (missing == 0) {
        printf("✓ CONVERSATION PATH IS COMPLETE!\n");
        printf("\nThe graph can route:\n");
        printf("  Mic → STT → Conversation Memory → LLM → Conversation Memory → TTS → Speaker\n");
    } else {
        printf("⚠ %d missing connections in conversation path\n", missing);
        printf("\nThe graph may need to learn these connections through UEL physics.\n");
        printf("Initial instinct patterns may be weak - graph will strengthen them.\n");
    }
}

int main(void) {
    Graph *g = melvin_open("/tmp/test_conv_path.m", 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    printf("Graph state:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    
    check_conversation_path(g);
    
    melvin_close(g);
    return 0;
}

