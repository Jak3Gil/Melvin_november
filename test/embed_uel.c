/* Embed UEL machine code into .m blob */
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

extern void melvin_tick(Graph *g);  /* From melvin_uel.c */

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <brain.m>\n", argv[0]);
        return 1;
    }
    
    const char *path = argv[1];
    
    /* Open brain file */
    int fd = open(path, O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    struct stat st;
    fstat(fd, &st);
    
    void *map = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    
    MelvinHeader *hdr = (MelvinHeader *)map;
    
    /* Validate */
    if (memcmp(hdr->magic, MELVIN_MAGIC, 4) != 0) {
        fprintf(stderr, "Invalid magic\n");
        munmap(map, st.st_size);
        close(fd);
        return 1;
    }
    
    /* Get blob region */
    uint8_t *blob = (uint8_t *)((char *)map + hdr->blob_offset);
    
    /* For now, we'll link melvin_tick directly */
    /* In a real system, you'd extract the machine code from melvin_uel.o */
    /* and copy it into the blob */
    
    /* Set tick_entry_offset to 0 (start of blob) */
    /* In practice, you'd copy the machine code and set offset to where it starts */
    hdr->tick_entry_offset = 0;
    
    printf("Set tick_entry_offset = 0\n");
    printf("Note: In production, copy melvin_tick machine code to blob[0]\n");
    printf("      For now, linking melvin_tick directly works\n");
    
    msync(map, st.st_size, MS_SYNC);
    munmap(map, st.st_size);
    close(fd);
    
    return 0;
}

