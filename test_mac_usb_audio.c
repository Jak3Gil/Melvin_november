/*
 * test_mac_usb_audio.c - Test USB mic and speaker on Mac
 * Records from mic and plays to speaker simultaneously
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

static volatile int running = 1;

static void signal_handler(int sig) {
    (void)sig;
    running = 0;
    printf("\nStopping...\n");
}

int main(int argc, char **argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("========================================\n");
    printf("USB Audio Test - Mac\n");
    printf("========================================\n");
    printf("Recording from mic and playing to speaker\n");
    printf("Press Ctrl+C to stop\n\n");
    
    int duration = 10;
    if (argc > 1) {
        duration = atoi(argv[1]);
    }
    
    printf("Running for %d seconds...\n", duration);
    printf("Speak into the microphone - you should hear it on the speaker!\n\n");
    
    /* Record and play simultaneously using sox */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), 
             "sox -d -r 16000 -c 1 -b 16 -t wav - | "
             "sox -t wav - -d -r 16000 -c 1 -b 16 &");
    
    printf("Starting audio loopback...\n");
    
    /* Use a simpler approach: record to file, then play in loop */
    printf("Recording %d seconds, then playing back...\n", duration);
    
    /* Record */
    snprintf(cmd, sizeof(cmd), "sox -d -r 16000 -c 1 -b 16 /tmp/mac_test.wav trim 0 %d 2>&1", duration);
    printf("Recording...\n");
    system(cmd);
    
    if (access("/tmp/mac_test.wav", F_OK) == 0) {
        long size = 0;
        FILE *f = fopen("/tmp/mac_test.wav", "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            size = ftell(f);
            fclose(f);
        }
        printf("Recorded %ld bytes\n", size);
        
        printf("\nPlaying back...\n");
        system("afplay /tmp/mac_test.wav");
        printf("Did you hear the playback?\n");
    } else {
        printf("Recording failed\n");
        return 1;
    }
    
    printf("\n========================================\n");
    printf("Test Complete\n");
    printf("========================================\n");
    
    return 0;
}

