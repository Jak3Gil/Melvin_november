/*
 * melvin_hardware_audio.c - Hardware audio I/O for Melvin
 * 
 * Connects USB microphone and speaker to Melvin's soft structure:
 * - Microphone → Port 0 (raw audio bytes)
 * - Speaker ← Port 100 (audio output)
 * 
 * Uses ALSA (preferred) or PulseAudio fallback on Linux/Jetson.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>

#ifdef __linux__
#include <alsa/asoundlib.h>
#endif

/* Audio configuration */
#define AUDIO_SAMPLE_RATE 48000  /* 48kHz - USB device requirement */
#define AUDIO_CHANNELS 2         /* Stereo (USB device requirement) */
#ifdef __linux__
#define AUDIO_FORMAT SND_PCM_FORMAT_S16_LE  /* 16-bit signed little-endian */
#else
#define AUDIO_FORMAT 0  /* Placeholder for non-Linux */
#endif
#define AUDIO_BUFFER_SIZE 1024     /* Bytes per read/write */
#define AUDIO_PORT_INPUT 0          /* Port for audio input */
#define AUDIO_PORT_OUTPUT 100       /* Port for audio output */
#define AUDIO_ACTIVATION_THRESHOLD 0.1f  /* Output port activation threshold (lowered for initial learning) */

typedef struct {
    Graph *g;
    bool running;
    pthread_t reader_thread;
    pthread_t writer_thread;
    
#ifdef __linux__
    snd_pcm_t *capture_handle;  /* Microphone */
    snd_pcm_t *playback_handle; /* Speaker */
#endif
    
    /* Statistics */
    uint64_t audio_bytes_read;
    uint64_t audio_bytes_written;
    
    /* Direct echo buffer - bypasses graph for immediate audio passthrough */
    uint8_t echo_buffer[AUDIO_BUFFER_SIZE];
    size_t echo_buffer_size;
    pthread_mutex_t echo_mutex;
} AudioHardware;

static AudioHardware *audio_hw = NULL;

/* Initialize ALSA audio */
#ifdef __linux__
static int init_alsa_capture(snd_pcm_t **handle, const char *device) {
    int err;
    snd_pcm_hw_params_t *hw_params;
    
    if ((err = snd_pcm_open(handle, device ? device : "default", SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "Cannot open audio device %s: %s\n", device ? device : "default", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0) {
        fprintf(stderr, "Cannot allocate hardware parameter structure: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_any(*handle, hw_params)) < 0) {
        fprintf(stderr, "Cannot initialize hardware parameter structure: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_set_access(*handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf(stderr, "Cannot set access type: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_set_format(*handle, hw_params, AUDIO_FORMAT)) < 0) {
        fprintf(stderr, "Cannot set sample format: %s\n", snd_strerror(err));
        return -1;
    }
    
    unsigned int rate = AUDIO_SAMPLE_RATE;
    if ((err = snd_pcm_hw_params_set_rate_near(*handle, hw_params, &rate, 0)) < 0) {
        fprintf(stderr, "Cannot set sample rate: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_set_channels(*handle, hw_params, AUDIO_CHANNELS)) < 0) {
        fprintf(stderr, "Cannot set channel count: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params(*handle, hw_params)) < 0) {
        fprintf(stderr, "Cannot set parameters: %s\n", snd_strerror(err));
        return -1;
    }
    
    snd_pcm_hw_params_free(hw_params);
    
    if ((err = snd_pcm_prepare(*handle)) < 0) {
        fprintf(stderr, "Cannot prepare audio interface: %s\n", snd_strerror(err));
        return -1;
    }
    
    return 0;
}

static int init_alsa_playback(snd_pcm_t **handle, const char *device) {
    int err;
    snd_pcm_hw_params_t *hw_params;
    
    if ((err = snd_pcm_open(handle, device ? device : "default", SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        fprintf(stderr, "Cannot open audio device %s: %s\n", device ? device : "default", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0) {
        fprintf(stderr, "Cannot allocate hardware parameter structure: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_any(*handle, hw_params)) < 0) {
        fprintf(stderr, "Cannot initialize hardware parameter structure: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_set_access(*handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf(stderr, "Cannot set access type: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_set_format(*handle, hw_params, AUDIO_FORMAT)) < 0) {
        fprintf(stderr, "Cannot set sample format: %s\n", snd_strerror(err));
        return -1;
    }
    
    unsigned int rate = AUDIO_SAMPLE_RATE;
    if ((err = snd_pcm_hw_params_set_rate_near(*handle, hw_params, &rate, 0)) < 0) {
        fprintf(stderr, "Cannot set sample rate: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params_set_channels(*handle, hw_params, AUDIO_CHANNELS)) < 0) {
        fprintf(stderr, "Cannot set channel count: %s\n", snd_strerror(err));
        return -1;
    }
    
    if ((err = snd_pcm_hw_params(*handle, hw_params)) < 0) {
        fprintf(stderr, "Cannot set parameters: %s\n", snd_strerror(err));
        return -1;
    }
    
    snd_pcm_hw_params_free(hw_params);
    
    if ((err = snd_pcm_prepare(*handle)) < 0) {
        fprintf(stderr, "Cannot prepare audio interface: %s\n", snd_strerror(err));
        return -1;
    }
    
    return 0;
}
#endif

/* Audio reader thread - reads from microphone and feeds to graph */
/* EVENT-DRIVEN: Hardware feeds bytes, graph processes them */
/* GRACEFUL ERROR HANDLING: If device fails, thread continues (graph learns) */
static void *audio_reader_thread(void *arg) {
    AudioHardware *hw = (AudioHardware *)arg;
    uint8_t buffer[AUDIO_BUFFER_SIZE];
    int consecutive_errors = 0;
    const int MAX_CONSECUTIVE_ERRORS = 10;  /* After 10 errors, pause and retry */
    
    printf("Audio reader thread started\n");
    
#ifdef __linux__
    if (!hw->capture_handle) {
        fprintf(stderr, "Audio capture not initialized\n");
        return NULL;
    }
    
    while (hw->running) {
        snd_pcm_sframes_t frames_read = snd_pcm_readi(hw->capture_handle, buffer, 
                                                       AUDIO_BUFFER_SIZE / (2 * AUDIO_CHANNELS));  /* 2 bytes per sample * channels */
        
        if (frames_read < 0) {
            if (frames_read == -EPIPE) {
                /* Overrun occurred */
                snd_pcm_prepare(hw->capture_handle);
                continue;
            }
            fprintf(stderr, "Error reading audio: %s\n", snd_strerror(frames_read));
            usleep(10000);  /* Wait 10ms before retry */
            continue;
        }
        
        if (frames_read > 0) {
            size_t bytes_read = (size_t)frames_read * 2 * AUDIO_CHANNELS;  /* 2 bytes per sample * channels */
            
            /* Store audio in echo buffer for direct passthrough */
            pthread_mutex_lock(&hw->echo_mutex);
            if (bytes_read > 0 && bytes_read <= AUDIO_BUFFER_SIZE) {
                memcpy(hw->echo_buffer, buffer, bytes_read);
                hw->echo_buffer_size = bytes_read;
            }
            pthread_mutex_unlock(&hw->echo_mutex);
            
            /* Feed audio bytes to graph via port 0 */
            for (size_t i = 0; i < bytes_read && i < AUDIO_BUFFER_SIZE; i++) {
                float energy = 0.3f;  /* Higher energy for audio input to help learning */
                melvin_feed_byte(hw->g, AUDIO_PORT_INPUT, buffer[i], energy);
                hw->audio_bytes_read++;
            }
            
            /* Also feed to working memory (200-209) and STT gateway input (300) for tool invocation */
            if (bytes_read > 0 && hw->g->node_count > 310) {
                /* Feed samples to working memory (200-209) */
                for (size_t i = 0; i < bytes_read && i < 10; i++) {
                    melvin_feed_byte(hw->g, 200 + (i % 10), buffer[i], 0.2f);
                }
                /* Activate STT gateway input (300) to trigger tool */
                melvin_feed_byte(hw->g, 300, buffer[0], 0.4f); /* Higher energy to trigger tool */
            }
            
            /* Also seed weak connection to output port to help graph learn routing */
            if (bytes_read > 0) {
                /* Feed a sample to output port with lower energy to suggest routing */
                melvin_feed_byte(hw->g, AUDIO_PORT_OUTPUT, buffer[0], 0.1f);
            }
            
            /* Don't call melvin_call_entry from thread - let main loop handle it */
            /* This prevents race conditions when multiple threads call it */
        }
    }
#else
    /* Fallback: simulate audio input for non-Linux systems */
    while (hw->running) {
        /* Generate test audio pattern */
        for (int i = 0; i < AUDIO_BUFFER_SIZE; i++) {
            buffer[i] = (uint8_t)(rand() % 256);
            float energy = 0.1f;
            melvin_feed_byte(hw->g, AUDIO_PORT_INPUT, buffer[i], energy);
            hw->audio_bytes_read++;
        }
        /* Don't call melvin_call_entry from thread - let main loop handle it */
        usleep(64000);  /* ~1 second at 16kHz (1024 samples / 16000 samples/sec) */
    }
#endif
    
    printf("Audio reader thread stopped\n");
    return NULL;
}

/* Audio writer thread - monitors output port and writes to speaker */
static void *audio_writer_thread(void *arg) {
    AudioHardware *hw = (AudioHardware *)arg;
    uint8_t buffer[AUDIO_BUFFER_SIZE];
    
    printf("Audio writer thread started\n");
    
#ifdef __linux__
    if (!hw->playback_handle) {
        fprintf(stderr, "Audio playback not initialized\n");
        return NULL;
    }
#endif
    
    while (hw->running) {
        /* Check if output port (100) is activated */
        float activation = melvin_get_activation(hw->g, AUDIO_PORT_OUTPUT);
        
        /* Also check input port - if there's input, echo it to output (helps learning) */
        float input_activation = melvin_get_activation(hw->g, AUDIO_PORT_INPUT);
        bool should_output = (activation > AUDIO_ACTIVATION_THRESHOLD) || (input_activation > 0.2f);
        
        if (should_output) {
            /* Graph wants to produce audio - read from output ports (100-109) or echo input */
            int count = 0;
            
            if (activation > AUDIO_ACTIVATION_THRESHOLD) {
                /* Use graph's output */
                for (int i = AUDIO_PORT_OUTPUT; i < AUDIO_PORT_OUTPUT + 10 && count < AUDIO_BUFFER_SIZE; i++) {
                    float a = melvin_get_activation(hw->g, i);
                    if (a > 0.1f) {  /* Lower threshold for output */
                        /* Convert activation to audio byte */
                        /* Activation is -1 to 1, convert to 0-255 for audio */
                        int16_t sample = (int16_t)(a * 32767.0f);
                        if (count + 1 < AUDIO_BUFFER_SIZE) {
                            buffer[count++] = (uint8_t)(sample & 0xFF);
                            buffer[count++] = (uint8_t)((sample >> 8) & 0xFF);
                        }
                    }
                }
            }
            
            /* If no graph output, use direct echo from mic input (raw audio passthrough) */
            if (count == 0) {
                /* Check echo buffer for direct audio passthrough - use raw bytes, no conversion */
                pthread_mutex_lock(&hw->echo_mutex);
                if (hw->echo_buffer_size > 0) {
                    size_t copy_size = (hw->echo_buffer_size < AUDIO_BUFFER_SIZE) ? 
                                       hw->echo_buffer_size : AUDIO_BUFFER_SIZE;
                    /* Copy raw audio bytes directly - no conversion needed */
                    memcpy(buffer, hw->echo_buffer, copy_size);
                    count = (int)copy_size;
                    /* Don't clear immediately - let it persist for a few cycles for smoother echo */
                }
                pthread_mutex_unlock(&hw->echo_mutex);
            }
            
            /* Don't use graph activation for echo - it produces static */
            /* Only use direct raw audio passthrough from echo buffer */
            
            if (count > 0) {
#ifdef __linux__
                /* Ensure we have at least a few samples */
                if (count < 64) {
                    /* Pad with silence to avoid underruns */
                    while (count < 64 && count < AUDIO_BUFFER_SIZE) {
                        buffer[count++] = 0;
                        buffer[count++] = 0;
                    }
                }
                
                snd_pcm_sframes_t frames_written = snd_pcm_writei(hw->playback_handle, buffer, count / 2);
                if (frames_written < 0) {
                    if (frames_written == -EPIPE) {
                        /* Underrun occurred */
                        snd_pcm_prepare(hw->playback_handle);
                        /* Retry */
                        frames_written = snd_pcm_writei(hw->playback_handle, buffer, count / 2);
                    }
                    if (frames_written < 0) {
                        fprintf(stderr, "Error writing audio: %s\n", snd_strerror(frames_written));
                    } else {
                        hw->audio_bytes_written += (size_t)frames_written * 2;
                    }
                } else {
                    hw->audio_bytes_written += (size_t)frames_written * 2 * AUDIO_CHANNELS;
                    
                    /* Provide positive feedback */
                    melvin_feed_byte(hw->g, 30, 1, 0.5f);  /* Positive feedback node */
                }
#else
                /* Fallback: just count bytes */
                hw->audio_bytes_written += count;
                melvin_feed_byte(hw->g, 30, 1, 0.5f);
#endif
            }
        }
        
        usleep(5000);  /* Check every 5ms for faster echo response */
    }
    
    printf("Audio writer thread stopped\n");
    return NULL;
}

/* Initialize audio hardware */
int melvin_hardware_audio_init(Graph *g, const char *capture_device, const char *playback_device) {
    if (!g) return -1;
    
    audio_hw = calloc(1, sizeof(AudioHardware));
    if (!audio_hw) return -1;
    
    audio_hw->g = g;
    audio_hw->running = true;
    audio_hw->echo_buffer_size = 0;
    pthread_mutex_init(&audio_hw->echo_mutex, NULL);
    
#ifdef __linux__
    /* Initialize ALSA capture (microphone) */
    if (init_alsa_capture(&audio_hw->capture_handle, capture_device) < 0) {
        fprintf(stderr, "Warning: Failed to initialize audio capture, continuing without mic\n");
        audio_hw->capture_handle = NULL;
    }
    
    /* Initialize ALSA playback (speaker) */
    if (init_alsa_playback(&audio_hw->playback_handle, playback_device) < 0) {
        fprintf(stderr, "Warning: Failed to initialize audio playback, continuing without speaker\n");
        audio_hw->playback_handle = NULL;
    }
    
    if (!audio_hw->capture_handle && !audio_hw->playback_handle) {
        fprintf(stderr, "Warning: No audio hardware available, using simulation mode\n");
    }
#endif
    
    /* Start reader thread (microphone → graph) */
    if (pthread_create(&audio_hw->reader_thread, NULL, audio_reader_thread, audio_hw) != 0) {
        fprintf(stderr, "Failed to create audio reader thread\n");
        free(audio_hw);
        audio_hw = NULL;
        return -1;
    }
    
    /* Start writer thread (graph → speaker) */
    if (pthread_create(&audio_hw->writer_thread, NULL, audio_writer_thread, audio_hw) != 0) {
        fprintf(stderr, "Failed to create audio writer thread\n");
        audio_hw->running = false;
        pthread_join(audio_hw->reader_thread, NULL);
        free(audio_hw);
        audio_hw = NULL;
        return -1;
    }
    
    printf("Audio hardware initialized\n");
    printf("  Capture device: %s\n", capture_device ? capture_device : "default");
    printf("  Playback device: %s\n", playback_device ? playback_device : "default");
    printf("  Input port: %d\n", AUDIO_PORT_INPUT);
    printf("  Output port: %d\n", AUDIO_PORT_OUTPUT);
    
    return 0;
}

/* Shutdown audio hardware */
void melvin_hardware_audio_shutdown(void) {
    if (!audio_hw) return;
    
    audio_hw->running = false;
    pthread_mutex_destroy(&audio_hw->echo_mutex);
    
    pthread_join(audio_hw->reader_thread, NULL);
    pthread_join(audio_hw->writer_thread, NULL);
    
#ifdef __linux__
    if (audio_hw->capture_handle) {
        snd_pcm_close(audio_hw->capture_handle);
    }
    if (audio_hw->playback_handle) {
        snd_pcm_close(audio_hw->playback_handle);
    }
#endif
    
    printf("Audio hardware shutdown\n");
    printf("  Bytes read: %llu\n", (unsigned long long)audio_hw->audio_bytes_read);
    printf("  Bytes written: %llu\n", (unsigned long long)audio_hw->audio_bytes_written);
    
    free(audio_hw);
    audio_hw = NULL;
}

/* Get statistics */
void melvin_hardware_audio_stats(uint64_t *bytes_read, uint64_t *bytes_written) {
    if (!audio_hw) {
        if (bytes_read) *bytes_read = 0;
        if (bytes_written) *bytes_written = 0;
        return;
    }
    
    if (bytes_read) *bytes_read = audio_hw->audio_bytes_read;
    if (bytes_written) *bytes_written = audio_hw->audio_bytes_written;
}

