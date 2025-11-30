/*
 * melvin_tools.h - Tool interface for pattern generation
 */

#ifndef MELVIN_TOOLS_H
#define MELVIN_TOOLS_H

#include <stdint.h>
#include <stddef.h>

/* LLM: text → text (local Ollama) */
int melvin_tool_llm_generate(const uint8_t *prompt, size_t prompt_len,
                            uint8_t **response, size_t *response_len);

/* Vision: image bytes → labels (local ONNX/PyTorch) */
int melvin_tool_vision_identify(const uint8_t *image_bytes, size_t image_len,
                                uint8_t **labels, size_t *labels_len);

/* Audio STT: audio bytes → text (local Whisper/Vosk) */
int melvin_tool_audio_stt(const uint8_t *audio_bytes, size_t audio_len,
                         uint8_t **text, size_t *text_len);

/* Audio TTS: text → audio bytes (local piper/eSpeak) */
int melvin_tool_audio_tts(const uint8_t *text, size_t text_len,
                          uint8_t **audio_bytes, size_t *audio_len);

#endif /* MELVIN_TOOLS_H */

