/*
 * melvin_tool_layer.c - Tool invocation layer (built on top of melvin.c substrate)
 * 
 * This layer watches the graph and invokes tools when tool gateway nodes activate.
 * Tools are external pattern generators - the graph decides when to use them.
 * 
 * This is NOT part of the substrate - it's a layer built on top of melvin.c
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* Tool invocation: watches graph for tool gateway activations and invokes tools */
void melvin_tool_layer_invoke(Graph *g) {
    if (!g || !g->hdr) return;
    
    MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
    if (!syscalls || g->node_count < 610) return;
    
    /* Scan tool gateway range (300-699) for activations */
    /* Graph decides which tools to use through activation patterns */
    for (uint32_t gateway_base = 300; gateway_base < 700; gateway_base += 100) {
        uint32_t input_node = gateway_base;      /* e.g., 300, 400, 500, 600 */
        uint32_t output_node = gateway_base + 10; /* e.g., 310, 410, 510, 610 */
        
        if (input_node >= g->node_count || output_node >= g->node_count) continue;
        
        float input_activation = fabsf(g->nodes[input_node].a);
        /* NaN check */
        if (input_activation != input_activation) input_activation = 0.0f;
        
        /* Dynamic threshold based on graph state */
        float threshold = (g->avg_activation > 0.001f) ? (g->avg_activation * 1.5f) : 0.05f;
        
        if (input_activation <= threshold) continue;
        
        /* Determine which tool to call based on gateway range */
        /* STT: 300-309 input, 310-319 output */
        if (gateway_base == 300 && syscalls->sys_audio_stt) {
            static uint32_t stt_call_count = 0;
            /* Collect audio data from working memory (200-209) */
            static uint8_t audio_buffer[48000 * 2 * 2]; /* 2 seconds @ 48kHz stereo */
            size_t audio_len = 0;
            
            for (uint32_t i = 200; i < 210 && audio_len < sizeof(audio_buffer) - 256; i++) {
                if (i < g->node_count && g->nodes[i].a > 0.05f) {
                    audio_buffer[audio_len++] = g->nodes[i].byte;
                }
            }
            
            if (audio_len > 16000) { /* Minimum audio length */
                stt_call_count++;
                if (stt_call_count <= 3) {
                    printf("[TOOL] STT called (activation=%.3f, audio_len=%zu)\n", input_activation, audio_len);
                    fflush(stdout);
                }
                
                uint8_t *text = NULL;
                size_t text_len = 0;
                int result = syscalls->sys_audio_stt(audio_buffer, audio_len, &text, &text_len);
                if (result == 0 && text && text_len > 0) {
                    if (stt_call_count <= 3) {
                        printf("[TOOL] STT output: %.*s\n", (int)(text_len < 100 ? text_len : 100), text);
                        fflush(stdout);
                    }
                    /* Feed STT output into graph - becomes graph structure */
                    for (size_t i = 0; i < text_len && i < 256; i++) {
                        melvin_feed_byte(g, output_node, text[i], 0.8f); /* Tool output */
                        melvin_feed_byte(g, 200 + (i % 10), text[i], 0.5f); /* Working memory */
                    }
                    free(text);
                }
            }
        }
        /* Vision: 400-409 input, 410-419 output */
        else if (gateway_base == 400 && syscalls->sys_vision_identify) {
            static uint32_t vision_call_count = 0;
            /* Collect image data from working memory (201-210) */
            static uint8_t image_buffer[640 * 360 * 3];
            size_t image_len = 0;
            
            for (uint32_t i = 201; i < 211 && image_len < sizeof(image_buffer) - 256; i++) {
                if (i < g->node_count && g->nodes[i].a > 0.1f) {
                    image_buffer[image_len++] = g->nodes[i].byte;
                }
            }
            
            if (image_len > 10000) {
                vision_call_count++;
                if (vision_call_count <= 3) {
                    printf("[TOOL] Vision called (activation=%.3f, image_len=%zu)\n", input_activation, image_len);
                    fflush(stdout);
                }
                
                uint8_t *labels = NULL;
                size_t labels_len = 0;
                int result = syscalls->sys_vision_identify(image_buffer, image_len, &labels, &labels_len);
                if (result == 0 && labels && labels_len > 0) {
                    if (vision_call_count <= 3) {
                        printf("[TOOL] Vision output: %.*s\n", (int)(labels_len < 100 ? labels_len : 100), labels);
                        fflush(stdout);
                    }
                    /* Feed Vision output into graph - becomes graph structure */
                    for (size_t i = 0; i < labels_len && i < 256; i++) {
                        melvin_feed_byte(g, output_node, labels[i], 0.8f); /* Tool output */
                        melvin_feed_byte(g, 201 + (i % 10), labels[i], 0.5f); /* Working memory */
                    }
                    free(labels);
                }
            }
        }
        /* LLM: 500-509 input, 510-519 output */
        else if (gateway_base == 500 && syscalls->sys_llm_generate) {
            static uint32_t llm_call_count = 0;
            /* Collect text from working memory (202-211) */
            static uint8_t prompt_buffer[4096];
            size_t prompt_len = 0;
            
            for (uint32_t i = 202; i < 212 && prompt_len < sizeof(prompt_buffer) - 1; i++) {
                if (i < g->node_count && g->nodes[i].a > 0.1f && 
                    g->nodes[i].byte >= 32 && g->nodes[i].byte < 127) {
                    prompt_buffer[prompt_len++] = g->nodes[i].byte;
                }
            }
            
            if (prompt_len > 10) {
                prompt_buffer[prompt_len] = '\0';
                llm_call_count++;
                if (llm_call_count <= 3) {
                    printf("[TOOL] LLM called (activation=%.3f, prompt_len=%zu)\n", input_activation, prompt_len);
                    fflush(stdout);
                }
                
                uint8_t *response = NULL;
                size_t response_len = 0;
                int result = syscalls->sys_llm_generate(prompt_buffer, prompt_len, &response, &response_len);
                if (result == 0 && response && response_len > 0) {
                    if (llm_call_count <= 3) {
                        printf("[TOOL] LLM output: %.*s\n", (int)(response_len < 100 ? response_len : 100), response);
                        fflush(stdout);
                    }
                    /* Feed LLM output into graph - becomes graph structure */
                    for (size_t i = 0; i < response_len && i < 512; i++) {
                        melvin_feed_byte(g, output_node, response[i], 0.8f); /* Tool output */
                        melvin_feed_byte(g, 202 + (i % 10), response[i], 0.5f); /* Working memory */
                    }
                    free(response);
                }
            }
        }
        /* TTS: 600-609 input, 610-619 output */
        else if (gateway_base == 600 && syscalls->sys_audio_tts) {
            static uint32_t tts_call_count = 0;
            /* Collect text from working memory (203-212) */
            static uint8_t tts_text_buffer[4096];
            size_t tts_text_len = 0;
            
            for (uint32_t i = 203; i < 213 && tts_text_len < sizeof(tts_text_buffer) - 1; i++) {
                if (i < g->node_count && g->nodes[i].a > 0.1f && 
                    g->nodes[i].byte >= 32 && g->nodes[i].byte < 127) {
                    tts_text_buffer[tts_text_len++] = g->nodes[i].byte;
                }
            }
            
            if (tts_text_len > 5) {
                tts_text_buffer[tts_text_len] = '\0';
                tts_call_count++;
                if (tts_call_count <= 3) {
                    printf("[TOOL] TTS called (activation=%.3f, text_len=%zu)\n", input_activation, tts_text_len);
                    fflush(stdout);
                }
                
                uint8_t *audio = NULL;
                size_t audio_len = 0;
                int result = syscalls->sys_audio_tts(tts_text_buffer, tts_text_len, &audio, &audio_len);
                if (result == 0 && audio && audio_len > 0) {
                    if (tts_call_count <= 3) {
                        printf("[TOOL] TTS output: %zu audio bytes\n", audio_len);
                        fflush(stdout);
                    }
                    /* Feed TTS output into graph - becomes graph structure */
                    for (size_t i = 0; i < audio_len && i < 48000 * 2; i++) {
                        melvin_feed_byte(g, output_node, audio[i], 0.8f); /* Tool output */
                        if (i < 48000 * 2) {
                            melvin_feed_byte(g, 120, audio[i], 0.6f); /* Audio output port */
                        }
                    }
                    free(audio);
                }
            }
        }
    }
}

