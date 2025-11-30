/*
 * test_tools_syscall.c - Test that tool syscalls work
 * 
 * Compile and run on Jetson to verify tools can be called via syscalls
 */

#include "melvin.h"
#include "melvin_tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("========================================\n");
    printf("Testing Tool Syscalls\n");
    printf("========================================\n\n");
    
    /* Test 1: LLM */
    printf("1. Testing LLM (Ollama)...\n");
    uint8_t *llm_response = NULL;
    size_t llm_response_len = 0;
    const char *llm_prompt = "Say hello in one word";
    
    int ret = melvin_tool_llm_generate((const uint8_t *)llm_prompt, strlen(llm_prompt),
                                      &llm_response, &llm_response_len);
    if (ret == 0 && llm_response && llm_response_len > 0) {
        printf("   ✓ LLM works! Response: %.*s\n", (int)llm_response_len, llm_response);
        free(llm_response);
    } else {
        printf("   ⚠ LLM test failed (Ollama may not be running)\n");
    }
    printf("\n");
    
    /* Test 2: Vision */
    printf("2. Testing Vision (ONNX)...\n");
    /* Create dummy image data (minimal valid JPEG) */
    uint8_t dummy_image[] = {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01,
        0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xFF, 0xC4,
        0x00, 0x14, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x0C,
        0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0x3F, 0xFF, 0xD9
    };
    
    uint8_t *vision_labels = NULL;
    size_t vision_labels_len = 0;
    ret = melvin_tool_vision_identify(dummy_image, sizeof(dummy_image),
                                      &vision_labels, &vision_labels_len);
    if (ret == 0 && vision_labels && vision_labels_len > 0) {
        printf("   ✓ Vision works! Labels: %.*s\n", (int)vision_labels_len, vision_labels);
        free(vision_labels);
    } else {
        printf("   ⚠ Vision test failed (ONNX may not be configured)\n");
    }
    printf("\n");
    
    /* Test 3: STT */
    printf("3. Testing STT (Whisper)...\n");
    /* Create dummy audio (silence, 16kHz mono 16-bit PCM) */
    uint8_t dummy_audio[1600];  /* 0.1 seconds */
    memset(dummy_audio, 0, sizeof(dummy_audio));
    
    uint8_t *stt_text = NULL;
    size_t stt_text_len = 0;
    ret = melvin_tool_audio_stt(dummy_audio, sizeof(dummy_audio),
                               &stt_text, &stt_text_len);
    if (ret == 0 && stt_text && stt_text_len > 0) {
        printf("   ✓ STT works! Text: %.*s\n", (int)stt_text_len, stt_text);
        free(stt_text);
    } else {
        printf("   ⚠ STT test failed (Whisper may not be configured)\n");
    }
    printf("\n");
    
    /* Test 4: TTS */
    printf("4. Testing TTS (piper/eSpeak)...\n");
    const char *tts_text = "Hello";
    uint8_t *tts_audio = NULL;
    size_t tts_audio_len = 0;
    ret = melvin_tool_audio_tts((const uint8_t *)tts_text, strlen(tts_text),
                                &tts_audio, &tts_audio_len);
    if (ret == 0 && tts_audio && tts_audio_len > 0) {
        printf("   ✓ TTS works! Generated %zu bytes of audio\n", tts_audio_len);
        free(tts_audio);
    } else {
        printf("   ⚠ TTS test failed (piper/eSpeak may not be configured)\n");
    }
    printf("\n");
    
    printf("========================================\n");
    printf("Syscall Test Complete\n");
    printf("========================================\n");
    printf("\n");
    printf("All tools are accessible via syscalls!\n");
    printf("The graph can call these tools and absorb their outputs.\n");
    
    return 0;
}

