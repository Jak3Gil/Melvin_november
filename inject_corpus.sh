#!/bin/bash
# Inject corpus + foundational knowledge into brain.m

cd /home/melvin/teachable_system

cat > inject_all.c << 'EOF'
#include "src/melvin.h"
#include <stdio.h>

int main() {
    Graph *b = melvin_open("responsive_brain.m", 10000, 50000, 131072);
    if (!b) return 1;
    
    int start = 0;
    for (uint64_t i = 840; i < b->node_count; i++) {
        if (b->nodes[i].pattern_data_offset > 0) start++;
    }
    
    printf("Injecting foundational knowledge...\n");
    
    // Math
    const char *k1 = "add subtract multiply divide modulo operations";
    for (const char *p = k1; *p; p++) melvin_feed_byte(b, 100, *p, 1.0f);
    for (int i = 0; i < 10; i++) melvin_call_entry(b);
    
    // GPU/CUDA
    const char *k2 = "cuda gpu kernel launch grid blocks threads memory allocate copy synchronize";
    for (const char *p = k2; *p; p++) melvin_feed_byte(b, 100, *p, 1.0f);
    for (int i = 0; i < 10; i++) melvin_call_entry(b);
    
    // Syscalls
    const char *k3 = "open read write close file socket connect send recv syscall";
    for (const char *p = k3; *p; p++) melvin_feed_byte(b, 100, *p, 1.0f);
    for (int i = 0; i < 10; i++) melvin_call_entry(b);
    
    // Compilation
    const char *k4 = "gcc compile link executable include define preprocessor linker";
    for (const char *p = k4; *p; p++) melvin_feed_byte(b, 100, *p, 1.0f);
    for (int i = 0; i < 10; i++) melvin_call_entry(b);
    
    int final = 0;
    for (uint64_t i = 840; i < b->node_count; i++) {
        if (b->nodes[i].pattern_data_offset > 0) final++;
    }
    
    printf("Patterns: %d -> %d (+%d)\n", start, final, final - start);
    printf("File size: %.2f MB\n", (double)b->hdr->file_size / (1024*1024));
    
    melvin_close(b);
    return 0;
}
EOF

gcc inject_all.c melvin.o -O2 -I. -lm -lpthread -o inject_all 2>&1 | grep -v warning
./inject_all
rm inject_all.c inject_all

echo ""
echo "✅ Foundational knowledge injected!"
echo ""

ls -lh responsive_brain.m

