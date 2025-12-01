/*
 * melvin_progress.c - Progress indicators for Melvin operations
 */

#include "melvin.h"
#include <stdio.h>
#include <stdint.h>

/* Progress callback function pointer */
static void (*progress_callback)(const char *message, float percent) = NULL;

/* Set progress callback */
void melvin_set_progress_callback(void (*callback)(const char *, float)) {
    progress_callback = callback;
}

/* Report progress */
void melvin_progress(const char *message, float percent) {
    if (progress_callback) {
        progress_callback(message, percent);
    } else {
        /* Default: simple progress bar */
        int bar_width = 50;
        int pos = (int)(bar_width * percent);
        
        printf("\r%s [", message);
        for (int i = 0; i < bar_width; i++) {
            if (i < pos) printf("=");
            else if (i == pos) printf(">");
            else printf(" ");
        }
        printf("] %3.0f%%", percent * 100);
        fflush(stdout);
        
        if (percent >= 1.0f) {
            printf("\n");
        }
    }
}

