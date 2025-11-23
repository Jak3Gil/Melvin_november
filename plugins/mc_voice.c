#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <fcntl.h>
#include <math.h>
#include <sys/stat.h>

// External helpers (from melvin.c)
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

// Voice state
static uint64_t melvin_voice_signature_node = UINT64_MAX;
static uint64_t phoneme_pattern_root = UINT64_MAX;
static int voice_initialized = 0;

// Melvin's voice signature (his "vocal cords" - unique characteristics)
// These are stored in the graph as patterns
static void initialize_melvin_voice(Brain *g) {
    if (voice_initialized) return;
    
    printf("[mc_voice] Initializing Melvin's voice signature...\n");
    
    // Create Melvin's voice signature pattern node
    melvin_voice_signature_node = alloc_node(g);
    if (melvin_voice_signature_node != UINT64_MAX) {
        Node *vs = &g->nodes[melvin_voice_signature_node];
        vs->kind = NODE_KIND_PATTERN_ROOT;
        vs->a = 0.7f;
        vs->bias = 0.5f;
        vs->value = 0x564F4943; // "VOIC" - voice signature
        
        // Store voice characteristics:
        // - Pitch baseline (Melvin's natural pitch)
        // - Rhythm patterns (Melvin's speaking tempo)
        // - Intonation patterns (Melvin's prosody)
        // - Vocal cord characteristics (formants, resonance)
        
        // Create characteristic nodes
        uint64_t pitch_node = alloc_node(g);
        if (pitch_node != UINT64_MAX) {
            Node *pn = &g->nodes[pitch_node];
            pn->kind = NODE_KIND_DATA;
            pn->a = 0.6f;
            pn->value = 220.0f; // Base pitch in Hz (A3 note - a default)
            add_edge(g, melvin_voice_signature_node, pitch_node, 1.0f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
        }
        
        uint64_t rhythm_node = alloc_node(g);
        if (rhythm_node != UINT64_MAX) {
            Node *rn = &g->nodes[rhythm_node];
            rn->kind = NODE_KIND_DATA;
            rn->a = 0.6f;
            rn->value = 150.0f; // Words per minute (default speaking rate)
            add_edge(g, melvin_voice_signature_node, rhythm_node, 1.0f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
        }
        
        printf("[mc_voice] Melvin's voice signature created (node %llu)\n", 
               (unsigned long long)melvin_voice_signature_node);
    }
    
    // Create phoneme pattern root (stores learned phoneme patterns)
    phoneme_pattern_root = alloc_node(g);
    if (phoneme_pattern_root != UINT64_MAX) {
        Node *ppr = &g->nodes[phoneme_pattern_root];
        ppr->kind = NODE_KIND_PATTERN_ROOT;
        ppr->a = 0.5f;
        ppr->value = 0x50484F4E; // "PHON" - phoneme patterns
        printf("[mc_voice] Phoneme pattern root created (node %llu)\n", 
               (unsigned long long)phoneme_pattern_root);
    }
    
    voice_initialized = 1;
}

// Helper: Extract phoneme pattern from audio (generalized content)
static uint64_t extract_phoneme_pattern(Brain *g, const char *phoneme_name) {
    if (phoneme_pattern_root == UINT64_MAX) return UINT64_MAX;
    
    // Hash phoneme name to find or create pattern node
    uint32_t hash = 0;
    size_t len = strlen(phoneme_name);
    for (size_t i = 0; i < len && i < 32; i++) {
        hash = hash * 31 + (unsigned char)phoneme_name[i];
    }
    
    // Search for existing phoneme pattern
    uint64_t n = g->header->num_nodes;
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &g->nodes[i];
        if (node->kind == NODE_KIND_DATA && (uint32_t)node->value == hash) {
            // Found existing phoneme pattern
            return i;
        }
    }
    
    // Create new phoneme pattern node
    uint64_t phoneme_node = alloc_node(g);
    if (phoneme_node != UINT64_MAX) {
        Node *pn = &g->nodes[phoneme_node];
        pn->kind = NODE_KIND_DATA;
        pn->a = 0.6f;
        pn->value = (float)hash;
        
        // Link to phoneme pattern root
        add_edge(g, phoneme_pattern_root, phoneme_node, 1.0f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
        
        printf("[mc_voice] Created phoneme pattern: %s (node %llu)\n", 
               phoneme_name, (unsigned long long)phoneme_node);
    }
    
    return phoneme_node;
}

// Helper: Create sequence of phonemes (words)
static void create_phoneme_sequence(Brain *g, const char *phonemes[], size_t count, uint64_t output_node) {
    uint64_t prev_phoneme = UINT64_MAX;
    
    for (size_t i = 0; i < count; i++) {
        uint64_t phoneme_node = extract_phoneme_pattern(g, phonemes[i]);
        if (phoneme_node != UINT64_MAX) {
            // Create sequence edge from previous phoneme
            if (prev_phoneme != UINT64_MAX) {
                add_edge(g, prev_phoneme, phoneme_node, 1.0f, EDGE_FLAG_SEQ);
            }
            
            // Link to output node
            add_edge(g, output_node, phoneme_node, 1.0f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
            
            prev_phoneme = phoneme_node;
        }
    }
}

// MC function: Process voice input (hear and learn patterns)
void mc_voice_in(Brain *g, uint64_t node_id) {
    static FILE *audio_fp = NULL;
    static char audio_path[512] = {0};
    
    // Initialize Melvin's voice
    if (!voice_initialized) {
        initialize_melvin_voice(g);
    }
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Look for audio file to process
    // In full implementation, this would read from microphone or audio file
    // For now, we'll process audio files from data/audio/
    
    if (!audio_fp) {
        // Try to open audio file
        static const char *audio_files[] = {
            "data/audio/input.wav",
            "data/audio/speech.wav",
            NULL
        };
        
        for (int i = 0; audio_files[i]; i++) {
            if (access(audio_files[i], F_OK) == 0) {
                audio_fp = fopen(audio_files[i], "rb");
                if (audio_fp) {
                    strncpy(audio_path, audio_files[i], sizeof(audio_path) - 1);
                    printf("[mc_voice] Processing audio file: %s\n", audio_path);
                    break;
                }
            }
        }
        
        if (!audio_fp) {
            // No audio file - check for audio input from stdin or microphone
            // For now, we'll create a placeholder pattern
            printf("[mc_voice] No audio file found. Ready for audio input.\n");
            g->nodes[node_id].a = 0.0f;
            return;
        }
    }
    
    // Read audio samples (simplified - would need actual audio processing)
    // In full implementation, would:
    // 1. Extract audio features (MFCC, pitch, etc.)
    // 2. Detect phonemes
    // 3. Extract patterns (separate content from voice characteristics)
    
    uint8_t audio_buffer[4096];
    size_t n = fread(audio_buffer, 1, sizeof(audio_buffer), audio_fp);
    
    if (n > 0) {
        // Create audio input node
        uint64_t audio_input_node = alloc_node(g);
        if (audio_input_node != UINT64_MAX) {
            Node *ain = &g->nodes[audio_input_node];
            ain->kind = NODE_KIND_DATA;
            ain->a = 0.8f;
            
            // Process audio samples (simplified - extract frequency patterns)
            // In real implementation, would use FFT or similar
            for (size_t i = 0; i < n && i < 256; i++) {
                uint8_t sample = audio_buffer[i];
                // Create frequency/amplitude nodes
                if (sample < g->header->num_nodes) {
                    Node *freq_node = &g->nodes[sample];
                    freq_node->a = 0.7f;
                    freq_node->value = (float)sample;
                    freq_node->kind = NODE_KIND_DATA;
                    
                    add_edge(g, audio_input_node, sample, 1.0f, EDGE_FLAG_SEQ);
                }
            }
            
            // Link to voice input node
            add_edge(g, node_id, audio_input_node, 1.0f, EDGE_FLAG_CONTROL);
            
            // Extract patterns (generalized - separate from voice characteristics)
            // This is where Melvin learns the CONTENT (phonemes) separately from VOICE
            
            printf("[mc_voice] Processed %zu audio samples, extracting patterns...\n", n);
        }
    }
    
    if (n < sizeof(audio_buffer)) {
        // End of file
        fclose(audio_fp);
        audio_fp = NULL;
        
        printf("[mc_voice] Finished processing audio file\n");
        g->nodes[node_id].a = 0.0f;
    }
}

// MC function: Generate voice output (speak with Melvin's voice)
void mc_voice_out(Brain *g, uint64_t node_id) {
    static uint64_t last_output_node = UINT64_MAX;
    
    // Initialize Melvin's voice
    if (!voice_initialized) {
        initialize_melvin_voice(g);
    }
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    // Find activated content nodes (phoneme patterns, text, etc.)
    // Then generate voice output using Melvin's voice signature
    
    // Look for activated phoneme patterns or text content
    uint64_t content_nodes[256] = {0};
    size_t content_count = 0;
    
    for (uint64_t i = 0; i < e_count && content_count < 256; i++) {
        Edge *e = &g->edges[i];
        if (e->dst == node_id && e->src < n) {
            Node *src = &g->nodes[e->src];
            // Look for DATA nodes with high activation (content to speak)
            if (src->kind == NODE_KIND_DATA && src->a > 0.5f) {
                content_nodes[content_count++] = e->src;
            }
        }
    }
    
    if (content_count > 0) {
        printf("[mc_voice] Generating voice output from %zu content nodes...\n", content_count);
        
        // Create voice output node
        uint64_t voice_output_node = alloc_node(g);
        if (voice_output_node != UINT64_MAX) {
            Node *vout = &g->nodes[voice_output_node];
            vout->kind = NODE_KIND_DATA;
            vout->a = 0.8f;
            vout->value = 0x564F5554; // "VOUT" - voice output
            
            // Combine content patterns with Melvin's voice signature
            // Link content to output
            for (size_t i = 0; i < content_count; i++) {
                add_edge(g, content_nodes[i], voice_output_node, 1.0f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
            }
            
            // Link Melvin's voice signature to output
            if (melvin_voice_signature_node != UINT64_MAX) {
                add_edge(g, melvin_voice_signature_node, voice_output_node, 1.0f, EDGE_FLAG_PATTERN | EDGE_FLAG_BIND);
                
                printf("[mc_voice] Combined content with Melvin's voice signature\n");
                printf("[mc_voice] Output: [Content] spoken with [Melvin's Voice]\n");
            }
            
            // In full implementation, this would:
            // 1. Take phoneme patterns from content
            // 2. Apply Melvin's voice characteristics (pitch, rhythm, intonation)
            // 3. Generate audio waveform
            // 4. Output to speaker/audio device
            
            last_output_node = voice_output_node;
        }
    } else {
        printf("[mc_voice] No content found to speak\n");
    }
    
    // Deactivate after processing
    g->nodes[node_id].a = 0.0f;
}

// MC function: Learn voice patterns from input (generalize and store)
void mc_voice_learn(Brain *g, uint64_t node_id) {
    static int learning = 0;
    
    // Initialize Melvin's voice
    if (!voice_initialized) {
        initialize_melvin_voice(g);
    }
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    if (learning) {
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    printf("[mc_voice] Learning voice patterns...\n");
    
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    // Find audio input nodes and extract patterns
    // Separate content (phonemes) from voice characteristics
    
    // Look for recent audio input nodes
    uint64_t audio_inputs[256] = {0};
    size_t input_count = 0;
    
    for (uint64_t i = 0; i < n && input_count < 256; i++) {
        Node *node = &g->nodes[i];
        if (node->kind == NODE_KIND_DATA && node->a > 0.3f) {
            // Check if connected to voice processing
            for (uint64_t j = 0; j < e_count; j++) {
                Edge *e = &g->edges[j];
                if (e->dst == node_id && e->src == i) {
                    audio_inputs[input_count++] = i;
                    break;
                }
            }
        }
    }
    
    // Extract patterns from audio inputs
    // Generalize phoneme patterns (content) separate from voice characteristics
    for (size_t i = 0; i < input_count; i++) {
        uint64_t audio_node = audio_inputs[i];
        
        // Extract phoneme sequence (generalized content)
        // In full implementation, would use speech recognition to get phonemes
        // For now, we'll create placeholder patterns
        
        // Example: Extract phoneme pattern from "hello"
        const char *phonemes[] = {"/h/", "/ɛ/", "/l/", "/oʊ/"};
        create_phoneme_sequence(g, phonemes, 4, audio_node);
        
        printf("[mc_voice] Extracted phoneme patterns from input\n");
    }
    
    // Create success node
    uint64_t success_node = alloc_node(g);
    if (success_node != UINT64_MAX) {
        Node *sn = &g->nodes[success_node];
        sn->kind = NODE_KIND_META;
        sn->a = 1.0f;
        sn->value = 0x564F4C52; // "VOLR" - voice learned
        add_edge(g, node_id, success_node, 1.0f, EDGE_FLAG_CONTROL);
    }
    
    learning = 1;
    g->nodes[node_id].a = 0.0f;
    printf("[mc_voice] Voice pattern learning complete\n");
}

