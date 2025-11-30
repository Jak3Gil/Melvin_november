/*
 * test_tools_conversation.c - Test Whisper STT and Piper TTS via graph tools
 * 
 * Demonstrates:
 * 1. Record audio from mic
 * 2. Use Whisper (STT) to convert speech to text
 * 3. Feed text into graph
 * 4. Graph processes (could use LLM here)
 * 5. Use Piper (TTS) to convert text to speech
 * 6. Play audio output
 */

#include "melvin.h"
#include "melvin_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <alsa/asoundlib.h>

/* Record audio from mic */
static int record_audio(uint8_t **audio_data, size_t *audio_len, int duration_sec) {
    snd_pcm_t *handle;
    int err;
    
    if ((err = snd_pcm_open(&handle, "hw:0,0", SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "Cannot open mic: %s\n", snd_strerror(err));
        return -1;
    }
    
    snd_pcm_hw_params_t *hw_params;
    snd_pcm_hw_params_malloc(&hw_params);
    snd_pcm_hw_params_any(handle, hw_params);
    snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(handle, hw_params, SND_PCM_FORMAT_S16_LE);
    unsigned int rate = 16000;
    snd_pcm_hw_params_set_rate_near(handle, hw_params, &rate, 0);
    snd_pcm_hw_params_set_channels(handle, hw_params, 1);
    snd_pcm_hw_params(handle, hw_params);
    snd_pcm_hw_params_free(hw_params);
    snd_pcm_prepare(handle);
    
    size_t samples = rate * duration_sec;
    size_t bytes = samples * 2;
    *audio_data = malloc(bytes);
    if (!*audio_data) {
        snd_pcm_close(handle);
        return -1;
    }
    
    printf("Recording %d seconds... speak now!\n", duration_sec);
    snd_pcm_sframes_t frames = snd_pcm_readi(handle, *audio_data, samples);
    snd_pcm_close(handle);
    
    if (frames > 0) {
        *audio_len = (size_t)frames * 2;
        printf("Recorded %zu bytes\n", *audio_len);
        return 0;
    }
    
    free(*audio_data);
    *audio_data = NULL;
    return -1;
}

/* Play audio to speaker */
static int play_audio(const uint8_t *audio_data, size_t audio_len) {
    snd_pcm_t *handle;
    int err;
    
    if ((err = snd_pcm_open(&handle, "hw:0,0", SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        fprintf(stderr, "Cannot open speaker: %s\n", snd_strerror(err));
        return -1;
    }
    
    snd_pcm_hw_params_t *hw_params;
    snd_pcm_hw_params_malloc(&hw_params);
    snd_pcm_hw_params_any(handle, hw_params);
    snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(handle, hw_params, SND_PCM_FORMAT_S16_LE);
    unsigned int rate = 16000;
    snd_pcm_hw_params_set_rate_near(handle, hw_params, &rate, 0);
    snd_pcm_hw_params_set_channels(handle, hw_params, 1);
    snd_pcm_hw_params(handle, hw_params);
    snd_pcm_hw_params_free(hw_params);
    snd_pcm_prepare(handle);
    
    printf("Playing audio...\n");
    snd_pcm_sframes_t frames = snd_pcm_writei(handle, audio_data, audio_len / 2);
    snd_pcm_drain(handle);
    snd_pcm_close(handle);
    
    return (frames > 0) ? 0 : -1;
}

int main(void) {
    printf("========================================\n");
    printf("Whisper STT + Piper TTS Test\n");
    printf("========================================\n\n");
    
    /* Step 1: Record audio from mic */
    uint8_t *audio_input = NULL;
    size_t audio_input_len = 0;
    
    if (record_audio(&audio_input, &audio_input_len, 3) < 0) {
        fprintf(stderr, "Failed to record audio\n");
        return 1;
    }
    
    printf("\n");
    
    /* Step 2: Use Whisper (STT) to convert speech to text */
    printf("Step 2: Converting speech to text with Whisper...\n");
    uint8_t *text = NULL;
    size_t text_len = 0;
    
    int stt_result = melvin_tool_audio_stt(audio_input, audio_input_len, &text, &text_len);
    
    if (stt_result == 0 && text && text_len > 0) {
        printf("✓ Whisper transcribed: \"%.*s\"\n", (int)text_len, text);
    } else {
        printf("⚠ Whisper failed, using fallback\n");
        const char *fallback = "Hello, this is a test";
        text_len = strlen(fallback);
        text = malloc(text_len + 1);
        if (text) {
            memcpy(text, fallback, text_len + 1);
        }
    }
    
    free(audio_input);
    audio_input = NULL;
    
    printf("\n");
    
    /* Step 3: Feed text into graph (simulate graph processing) */
    printf("Step 3: Feeding text into graph...\n");
    Graph *g = melvin_open("/tmp/test_tools_brain.m", 1000, 5000, 65536);
    if (g) {
        /* Feed text bytes into graph */
        for (size_t i = 0; i < text_len; i++) {
            melvin_feed_byte(g, 0, text[i], 0.3f);
        }
        melvin_call_entry(g);
        printf("✓ Text fed into graph (%zu bytes)\n", text_len);
        printf("  Nodes: %llu, Edges: %llu\n", 
               (unsigned long long)g->node_count,
               (unsigned long long)g->edge_count);
    } else {
        printf("⚠ Failed to open graph\n");
    }
    
    printf("\n");
    
    /* Step 4: Use Piper (TTS) to convert text to speech */
    printf("Step 4: Converting text to speech with Piper...\n");
    uint8_t *audio_output = NULL;
    size_t audio_output_len = 0;
    
    int tts_result = melvin_tool_audio_tts(text, text_len, &audio_output, &audio_output_len);
    
    if (tts_result == 0 && audio_output && audio_output_len > 0) {
        printf("✓ Piper generated %zu bytes of audio\n", audio_output_len);
    } else {
        printf("⚠ Piper failed\n");
        free(text);
        if (g) melvin_close(g);
        return 1;
    }
    
    free(text);
    text = NULL;
    
    printf("\n");
    
    /* Step 5: Play audio output */
    printf("Step 5: Playing audio output...\n");
    if (play_audio(audio_output, audio_output_len) == 0) {
        printf("✓ Audio played successfully\n");
    } else {
        printf("⚠ Failed to play audio\n");
    }
    
    free(audio_output);
    audio_output = NULL;
    
    if (g) {
        melvin_close(g);
    }
    
    printf("\n========================================\n");
    printf("Test Complete\n");
    printf("========================================\n");
    printf("\nFlow: Mic → Whisper (STT) → Graph → Piper (TTS) → Speaker\n");
    
    return 0;
}

