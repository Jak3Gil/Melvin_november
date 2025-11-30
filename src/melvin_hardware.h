/*
 * melvin_hardware.h - Hardware I/O interface for Melvin
 * 
 * Connects USB mic, speaker, and cameras to Melvin's soft structure.
 */

#ifndef MELVIN_HARDWARE_H
#define MELVIN_HARDWARE_H

#include "melvin.h"
#include <stdint.h>

/* Audio hardware */
int melvin_hardware_audio_init(Graph *g, const char *capture_device, const char *playback_device);
void melvin_hardware_audio_shutdown(void);
void melvin_hardware_audio_stats(uint64_t *bytes_read, uint64_t *bytes_written);

/* Video hardware */
int melvin_hardware_video_init(Graph *g, const char **camera_devices, int n_cameras);
void melvin_hardware_video_shutdown(void);
void melvin_hardware_video_stats(uint64_t *frames_read, uint64_t *frames_written);

#endif /* MELVIN_HARDWARE_H */

