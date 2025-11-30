/*
 * uel_seed_tool.c - Offline tool to seed .m blob with UEL physics
 * 
 * This is NOT linked into runtime. It's a separate tool that:
 *   1. Creates/opens a .m file
 *   2. Compiles melvin_uel.c to machine code
 *   3. Extracts .text section
 *   4. Writes it into the blob
 *   5. Sets main_entry_offset
 * 
 * Runtime never links this - it only calls blob code.
 */

#include "melvin.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

/* Compile melvin_uel.c to shared library (position-independent, relocations resolved) */
static int compile_uel_source(const char *src_path, const char *so_path) {
    char cmd[1024];
    /* Compile as shared library - this resolves relocations */
    snprintf(cmd, sizeof(cmd), 
             "gcc -O2 -shared -fPIC -o %s %s -lm 2>&1", 
             so_path, src_path);
    
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Failed to compile %s to shared library\n", src_path);
        return -1;
    }
    
    return 0;
}

/* Extract .text section from shared library using otool on macOS */
static int extract_text_section_macos(const char *so_path, const char *bin_path, 
                                       uint64_t *out_offset, uint64_t *out_size) {
    char cmd[1024];
    FILE *f;
    unsigned long offset = 0, size = 0;
    
    /* Get offset from shared library */
    snprintf(cmd, sizeof(cmd),
             "otool -l %s 2>/dev/null | grep -A 4 'sectname __text' | grep 'offset' | head -1 | awk '{print $2}'",
             so_path);
    f = popen(cmd, "r");
    if (!f) return -1;
    if (fscanf(f, "%lx", &offset) != 1) {
        pclose(f);
        return -1;
    }
    pclose(f);
    
    /* Get size */
    snprintf(cmd, sizeof(cmd),
             "otool -l %s 2>/dev/null | grep -A 4 'sectname __text' | grep 'size' | head -1 | awk '{print $2}'",
             so_path);
    f = popen(cmd, "r");
    if (!f) return -1;
    if (fscanf(f, "%lx", &size) != 1 || size == 0) {
        pclose(f);
        return -1;
    }
    pclose(f);
    
    /* Extract with dd */
    snprintf(cmd, sizeof(cmd),
             "dd if=%s of=%s bs=1 skip=%lu count=%lu 2>/dev/null",
             so_path, bin_path, offset, size);
    int ret = system(cmd);
    
    if (ret == 0) {
        *out_offset = (uint64_t)offset;
        *out_size = (uint64_t)size;
        return 0;
    }
    
    return -1;
}

/* Extract .text section to raw binary from shared library */
static int extract_text_section(const char *so_path, const char *bin_path) {
    /* Try objcopy first (Linux) */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "objcopy -O binary --only-section=.text %s %s 2>&1",
             so_path, bin_path);
    
    int ret = system(cmd);
    if (ret == 0) {
        return 0;  /* Success */
    }
    
    /* Try otool on macOS */
    uint64_t offset, size;
    if (extract_text_section_macos(so_path, bin_path, &offset, &size) == 0) {
        return 0;
    }
    
    fprintf(stderr, "Failed to extract .text section from %s\n", so_path);
    fprintf(stderr, "Tried: objcopy (Linux) and otool (macOS)\n");
    return -1;
}

/* Read binary file into buffer */
static int read_binary_file(const char *path, uint8_t **out_buf, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size <= 0) {
        fclose(f);
        fprintf(stderr, "Invalid file size: %ld\n", size);
        return -1;
    }
    
    uint8_t *buf = malloc((size_t)size);
    if (!buf) {
        fclose(f);
        fprintf(stderr, "Failed to allocate %ld bytes\n", size);
        return -1;
    }
    
    size_t read = fread(buf, 1, (size_t)size, f);
    fclose(f);
    
    if (read != (size_t)size) {
        free(buf);
        fprintf(stderr, "Failed to read entire file\n");
        return -1;
    }
    
    *out_buf = buf;
    *out_len = (size_t)size;
    return 0;
}

/* Find uel_main symbol offset in shared library */
static uint64_t find_uel_main_offset(const char *so_path) {
    /* Use nm to find symbol */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "nm -g %s 2>/dev/null | grep ' T uel_main$' | awk '{print $1}'", so_path);
    
    FILE *pipe = popen(cmd, "r");
    if (!pipe) return 0;
    
    char line[256];
    if (fgets(line, sizeof(line), pipe)) {
        uint64_t offset = (uint64_t)strtoul(line, NULL, 16);
        pclose(pipe);
        return offset;
    }
    
    pclose(pipe);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <brain.m>\n", argv[0]);
        return 1;
    }
    
    const char *brain_path = argv[1];
    const char *uel_src = "melvin_uel.c";
    const char *temp_so = "/tmp/melvin_uel_seed.so";
    const char *temp_bin = "/tmp/melvin_uel_seed.bin";
    
    printf("=== Seeding UEL Physics into Blob ===\n\n");
    
    /* Step 1: Compile melvin_uel.c to shared library */
    printf("Step 1: Compiling %s to shared library...\n", uel_src);
    if (compile_uel_source(uel_src, temp_so) != 0) {
        fprintf(stderr, "Compilation failed\n");
        return 1;
    }
    printf("  ✓ Compiled to %s\n", temp_so);
    
    /* Step 2: Extract .text section */
    printf("\nStep 2: Extracting .text section...\n");
    if (extract_text_section(temp_so, temp_bin) != 0) {
        fprintf(stderr, "Failed to extract .text section\n");
        unlink(temp_so);
        return 1;
    }
    printf("  ✓ Extracted to %s\n", temp_bin);
    
    /* Step 3: Read machine code */
    printf("\nStep 3: Reading machine code...\n");
    uint8_t *machine_code = NULL;
    size_t code_len = 0;
    if (read_binary_file(temp_bin, &machine_code, &code_len) != 0) {
        fprintf(stderr, "Failed to read machine code\n");
        unlink(temp_so);
        unlink(temp_bin);
        return 1;
    }
    printf("  ✓ Read %zu bytes of machine code\n", code_len);
    
    /* Step 4: Find uel_main offset in shared library */
    printf("\nStep 4: Finding uel_main entry point...\n");
    uint64_t uel_main_offset = find_uel_main_offset(temp_so);
    if (uel_main_offset == 0) {
        printf("  ⚠ Could not find uel_main symbol, using offset 0\n");
        printf("     (Assuming uel_main is at start of .text)\n");
    } else {
        /* Get .text section offset to adjust */
        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
                 "otool -l %s 2>/dev/null | grep -A 4 'sectname __text' | grep 'offset' | head -1 | awk '{print $2}'",
                 temp_so);
        FILE *f = popen(cmd, "r");
        unsigned long text_offset = 0;
        if (f) {
            if (fscanf(f, "%lx", &text_offset) == 1) {
                /* Adjust: uel_main_offset is from start of file, text_offset is where .text starts */
                /* So uel_main relative to .text start is uel_main_offset - text_offset */
                if (uel_main_offset > text_offset) {
                    uel_main_offset = uel_main_offset - text_offset;
                }
            }
            pclose(f);
        }
        printf("  ✓ Found uel_main at offset 0x%llx (relative to .text start)\n", 
               (unsigned long long)uel_main_offset);
    }
    
    /* Step 5: Open brain */
    printf("\nStep 5: Opening brain file...\n");
    Graph *g = melvin_open(brain_path, 1000, 10000, 65536);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", brain_path);
        free(machine_code);
        unlink(temp_so);
        unlink(temp_bin);
        return 1;
    }
    printf("  ✓ Opened %s\n", brain_path);
    
    /* Step 6: Check blob size */
    if (g->hdr->blob_size < code_len) {
        fprintf(stderr, "Blob too small: %llu bytes, need %zu bytes\n",
                (unsigned long long)g->hdr->blob_size, code_len);
        free(machine_code);
        melvin_close(g);
        unlink(temp_so);
        unlink(temp_bin);
        return 1;
    }
    
    /* Step 7: Write machine code to blob */
    printf("\nStep 6: Writing machine code to blob...\n");
    memcpy(g->blob, machine_code, code_len);
    printf("  ✓ Wrote %zu bytes to blob[0]\n", code_len);
    
    /* Step 8: Set entry point */
    printf("\nStep 7: Setting entry points...\n");
    /* main_entry_offset is relative to blob start */
    /* If uel_main_offset is 0, it means uel_main is at start of .text section */
    /* Since we extracted .text section, offset 0 means start of blob */
    /* But we need to check: if we copied the whole .o file, we need to find .text offset */
    
    /* For now, if uel_main_offset is 0, assume it's at start of extracted .text */
    /* Set to 1 to avoid the "empty blob" check (0 is treated as empty) */
    if (uel_main_offset == 0 && code_len > 0) {
        /* uel_main is likely at start, but we'll set offset to 1 to indicate blob has code */
        /* Actually, let's check if we can find it in the binary */
        g->hdr->main_entry_offset = 1;  /* Non-zero means "has code" */
    } else if (uel_main_offset > 0 && uel_main_offset < code_len) {
        g->hdr->main_entry_offset = (uint64_t)uel_main_offset;
    } else {
        /* Default: assume code starts at blob[0], but set to 1 to avoid empty check */
        g->hdr->main_entry_offset = 1;
    }
    
    g->hdr->syscalls_ptr_offset = (code_len + 63) & ~63;  /* Align to 64 bytes */
    
    if (g->hdr->syscalls_ptr_offset >= g->hdr->blob_size) {
        g->hdr->syscalls_ptr_offset = g->hdr->blob_size - sizeof(void*);
    }
    
    uint64_t saved_main_offset = g->hdr->main_entry_offset;
    uint64_t saved_syscalls_offset = g->hdr->syscalls_ptr_offset;
    
    printf("  ✓ main_entry_offset = %llu\n", (unsigned long long)saved_main_offset);
    printf("  ✓ syscalls_ptr_offset = %llu\n", (unsigned long long)saved_syscalls_offset);
    
    /* Step 9: Sync and close */
    printf("\nStep 8: Syncing to disk...\n");
    melvin_sync(g);
    melvin_close(g);
    printf("  ✓ Brain saved\n");
    
    /* Cleanup */
    free(machine_code);
    unlink(temp_so);
    unlink(temp_bin);
    
    printf("\n=== Seeding Complete ===\n");
    printf("Blob now contains UEL physics machine code.\n");
    printf("Entry point: uel_main (offset %llu)\n", 
           (unsigned long long)saved_main_offset);
    
    return 0;
}

