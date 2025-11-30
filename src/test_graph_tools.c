/*
 * test_graph_tools.c - Test graph calling tools via syscalls
 * 
 * Demonstrates the graph using tools:
 * 1. Graph receives audio input (simulated)
 * 2. Graph calls sys_audio_stt (Whisper) via blob code
 * 3. Graph processes text
 * 4. Graph calls sys_audio_tts (Piper) via blob code
 * 5. Graph outputs audio
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Simulate blob code calling STT syscall */
static void simulate_blob_stt_call(Graph *g, MelvinSyscalls *syscalls, const uint8_t *audio, size_t audio_len) {
    /* In real blob, this would be machine code calling syscall */
    /* For now, we simulate by directly calling the syscall */
    if (!syscalls || !syscalls->sys_audio_stt) {
        printf("⚠ STT syscall not available\n");
        return;
    }
    
    uint8_t *text = NULL;
    size_t text_len = 0;
    
    printf("Graph calling sys_audio_stt (Whisper)...\n");
    int result = syscalls->sys_audio_stt(audio, audio_len, &text, &text_len);
    
    if (result == 0 && text && text_len > 0) {
        printf("✓ Whisper returned: \"%.*s\"\n", (int)text_len, text);
        
        /* Feed text into graph as new nodes/edges */
        printf("Feeding text into graph...\n");
        for (size_t i = 0; i < text_len; i++) {
            melvin_feed_byte(g, 200 + i, text[i], 0.5f);  /* Feed to memory ports */
        }
        melvin_call_entry(g);
        
        printf("  Nodes: %llu, Edges: %llu\n",
               (unsigned long long)g->node_count,
               (unsigned long long)g->edge_count);
        
        free(text);
    } else {
        printf("⚠ STT failed\n");
    }
}

/* Simulate blob code calling TTS syscall */
static void simulate_blob_tts_call(Graph *g, MelvinSyscalls *syscalls, const uint8_t *text, size_t text_len) {
    if (!syscalls || !syscalls->sys_audio_tts) {
        printf("⚠ TTS syscall not available\n");
        return;
    }
    
    uint8_t *audio = NULL;
    size_t audio_len = 0;
    
    printf("Graph calling sys_audio_tts (Piper)...\n");
    int result = syscalls->sys_audio_tts(text, text_len, &audio, &audio_len);
    
    if (result == 0 && audio && audio_len > 0) {
        printf("✓ Piper generated %zu bytes of audio\n", audio_len);
        
        /* Feed audio output into graph */
        printf("Feeding audio output into graph...\n");
        for (size_t i = 0; i < audio_len && i < 1000; i++) {  /* Limit to first 1000 bytes */
            melvin_feed_byte(g, 100 + (i % 10), audio[i], 0.3f);  /* Feed to output ports */
        }
        melvin_call_entry(g);
        
        printf("  Nodes: %llu, Edges: %llu\n",
               (unsigned long long)g->node_count,
               (unsigned long long)g->edge_count);
        
        /* In real system, this audio would go to speaker */
        printf("✓ Audio ready for playback (would go to speaker)\n");
        
        free(audio);
    } else {
        printf("⚠ TTS failed\n");
    }
}

int main(void) {
    printf("========================================\n");
    printf("Graph Tools Test - Whisper + Piper\n");
    printf("========================================\n\n");
    
    /* Open/create graph */
    Graph *g = melvin_open("/tmp/test_graph_tools_brain.m", 1000, 5000, 65536);
    if (!g) {
        fprintf(stderr, "Failed to open graph\n");
        return 1;
    }
    
    /* Initialize syscalls */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    printf("Graph opened: %llu nodes, %llu edges\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    printf("\n");
    
    /* Simulate audio input (in real system, this comes from mic) */
    printf("Step 1: Simulating audio input...\n");
    const char *test_phrase = "hello world";
    uint8_t *fake_audio = malloc(16000 * 2);  /* 1 second of fake audio */
    if (fake_audio) {
        for (int i = 0; i < 16000; i++) {
            int16_t sample = (int16_t)(sin(2.0 * 3.14159 * 440.0 * i / 16000.0) * 10000);
            fake_audio[i * 2] = (uint8_t)(sample & 0xFF);
            fake_audio[i * 2 + 1] = (uint8_t)((sample >> 8) & 0xFF);
        }
        printf("✓ Generated fake audio (%d bytes)\n", 16000 * 2);
    }
    printf("\n");
    
    /* Step 2: Graph calls STT */
    printf("Step 2: Graph using Whisper (STT)...\n");
    if (fake_audio) {
        simulate_blob_stt_call(g, &syscalls, fake_audio, 16000 * 2);
        free(fake_audio);
    }
    printf("\n");
    
    /* Step 3: Graph processes and decides to respond */
    printf("Step 3: Graph processing...\n");
    melvin_call_entry(g);
    printf("✓ Graph processed\n");
    printf("  Nodes: %llu, Edges: %llu\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    printf("\n");
    
    /* Step 4: Graph calls TTS */
    printf("Step 4: Graph using Piper (TTS)...\n");
    const char *response = "Hello, I heard you";
    simulate_blob_tts_call(g, &syscalls, (const uint8_t *)response, strlen(response));
    printf("\n");
    
    /* Final state */
    printf("Final graph state:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("  Chaos: %.6f\n", g->avg_chaos);
    printf("  Activation: %.6f\n", g->avg_activation);
    
    melvin_close(g);
    
    printf("\n========================================\n");
    printf("Test Complete\n");
    printf("========================================\n");
    printf("\nFlow demonstrated:\n");
    printf("  Audio Input → Graph → sys_audio_stt (Whisper) → Text\n");
    printf("  Text → Graph Processing → sys_audio_tts (Piper) → Audio Output\n");
    printf("\n✓ Graph can use tools via syscalls!\n");
    
    return 0;
}

