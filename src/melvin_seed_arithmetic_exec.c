/*
 * melvin_seed_arithmetic_exec.c - Seed EXEC nodes for arithmetic operations
 * 
 * Creates EXEC nodes with actual CPU arithmetic operations (ADD, SUB, MUL, DIV, etc.)
 * These nodes perform real computation, not pattern matching.
 * 
 * Usage: melvin_seed_arithmetic_exec <melvin.m> [strength]
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ARM64 machine code stubs for arithmetic operations */
#if defined(__aarch64__) || defined(__arm64__)

/* ADD: x0 = x0 + x1, return x0 */
static const uint8_t ARM64_ADD[] = {
    0x00, 0x00, 0x01, 0x8b,  // add x0, x0, x1
    0xc0, 0x03, 0x5f, 0xd6   // ret
};

/* SUB: x0 = x0 - x1, return x0 */
static const uint8_t ARM64_SUB[] = {
    0x00, 0x00, 0x01, 0xcb,  // sub x0, x0, x1
    0xc0, 0x03, 0x5f, 0xd6   // ret
};

/* MUL: x0 = x0 * x1, return x0 */
static const uint8_t ARM64_MUL[] = {
    0x00, 0x7c, 0x01, 0x9b,  // mul x0, x0, x1
    0xc0, 0x03, 0x5f, 0xd6   // ret
};

/* AND: x0 = x0 & x1, return x0 */
static const uint8_t ARM64_AND[] = {
    0x00, 0x00, 0x01, 0x8a,  // and x0, x0, x1
    0xc0, 0x03, 0x5f, 0xd6   // ret
};

/* OR: x0 = x0 | x1, return x0 */
static const uint8_t ARM64_OR[] = {
    0x00, 0x00, 0x01, 0xaa,  // orr x0, x0, x1
    0xc0, 0x03, 0x5f, 0xd6   // ret
};

/* XOR: x0 = x0 ^ x1, return x0 */
static const uint8_t ARM64_XOR[] = {
    0x00, 0x00, 0x01, 0xca,  // eor x0, x0, x1
    0xc0, 0x03, 0x5f, 0xd6   // ret
};

#else
/* x86_64 fallback (if needed) */
static const uint8_t X86_64_ADD[] = {
    0x48, 0x01, 0xf8,  // add rax, rdi
    0xc3               // ret
};
static const uint8_t X86_64_SUB[] = {
    0x48, 0x29, 0xf8,  // sub rax, rdi
    0xc3               // ret
};
static const uint8_t X86_64_MUL[] = {
    0x48, 0x0f, 0xaf, 0xc7,  // imul rax, rdi
    0xc3                     // ret
};
static const uint8_t X86_64_AND[] = {
    0x48, 0x21, 0xf8,  // and rax, rdi
    0xc3               // ret
};
static const uint8_t X86_64_OR[] = {
    0x48, 0x09, 0xf8,  // or rax, rdi
    0xc3               // ret
};
static const uint8_t X86_64_XOR[] = {
    0x48, 0x31, 0xf8,  // xor rax, rdi
    0xc3               // ret
};
#endif

/* Write machine code to blob and create EXEC node */
static uint32_t create_arithmetic_exec_node(Graph *g, const uint8_t *code, size_t code_len, 
                                            uint32_t node_id, float threshold_ratio) {
    if (!g || !g->hdr || !code || code_len == 0) return UINT32_MAX;
    
    /* Find space in blob */
    uint64_t offset = g->hdr->main_entry_offset;
    if (offset == 0) {
        offset = 256;  /* Start after header */
    } else {
        offset += 256;  /* After existing code */
    }
    
    /* Check if we have space */
    if (offset + (uint64_t)code_len > g->hdr->blob_size) {
        fprintf(stderr, "Warning: Not enough blob space for code\n");
        return UINT32_MAX;
    }
    
    /* Write code to blob */
    memcpy(g->blob + offset, code, code_len);
    
    /* Ensure node exists */
    if (node_id >= g->node_count) {
        /* Would need to grow graph - simplified for now */
        fprintf(stderr, "Warning: Node ID %u exceeds graph size\n", node_id);
        return UINT32_MAX;
    }
    
    /* Create EXEC node */
    return melvin_create_exec_node(g, node_id, offset, threshold_ratio);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <melvin.m file> [threshold_ratio]\n", argv[0]);
        fprintf(stderr, "  threshold_ratio: EXEC activation threshold (default: 1.0)\n");
        return 1;
    }
    
    const char *path = argv[1];
    float threshold_ratio = (argc >= 3) ? (float)atof(argv[2]) : 1.0f;
    
    /* Open existing .m file */
    Graph *g = melvin_open(path, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", path);
        return 1;
    }
    
    printf("Seeding arithmetic EXEC nodes into %s...\n", path);
    printf("Graph has %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    printf("Threshold ratio: %.2f\n\n", threshold_ratio);
    
    /* Reserve node IDs for arithmetic EXEC nodes (2000-2099) */
    uint32_t EXEC_ADD = 2000;
    uint32_t EXEC_SUB = 2001;
    uint32_t EXEC_MUL = 2002;
    uint32_t EXEC_DIV = 2003;  /* Would need more complex code */
    uint32_t EXEC_AND = 2004;
    uint32_t EXEC_OR = 2005;
    uint32_t EXEC_XOR = 2006;
    
    size_t created = 0;
    
#if defined(__aarch64__) || defined(__arm64__)
    printf("Creating ARM64 arithmetic EXEC nodes...\n");
    
    if (create_arithmetic_exec_node(g, ARM64_ADD, sizeof(ARM64_ADD), EXEC_ADD, threshold_ratio) != UINT32_MAX) {
        printf("  ✓ ADD EXEC node: %u\n", EXEC_ADD);
        created++;
    }
    
    if (create_arithmetic_exec_node(g, ARM64_SUB, sizeof(ARM64_SUB), EXEC_SUB, threshold_ratio) != UINT32_MAX) {
        printf("  ✓ SUB EXEC node: %u\n", EXEC_SUB);
        created++;
    }
    
    if (create_arithmetic_exec_node(g, ARM64_MUL, sizeof(ARM64_MUL), EXEC_MUL, threshold_ratio) != UINT32_MAX) {
        printf("  ✓ MUL EXEC node: %u\n", EXEC_MUL);
        created++;
    }
    
    if (create_arithmetic_exec_node(g, ARM64_AND, sizeof(ARM64_AND), EXEC_AND, threshold_ratio) != UINT32_MAX) {
        printf("  ✓ AND EXEC node: %u\n", EXEC_AND);
        created++;
    }
    
    if (create_arithmetic_exec_node(g, ARM64_OR, sizeof(ARM64_OR), EXEC_OR, threshold_ratio) != UINT32_MAX) {
        printf("  ✓ OR EXEC node: %u\n", EXEC_OR);
        created++;
    }
    
    if (create_arithmetic_exec_node(g, ARM64_XOR, sizeof(ARM64_XOR), EXEC_XOR, threshold_ratio) != UINT32_MAX) {
        printf("  ✓ XOR EXEC node: %u\n", EXEC_XOR);
        created++;
    }
#else
    printf("Warning: Not ARM64 - arithmetic EXEC nodes may not work correctly\n");
    printf("  (Code stubs are ARM64-specific)\n");
#endif
    
    printf("\nCreated %zu arithmetic EXEC nodes\n", created);
    
    /* Sync to disk */
    melvin_sync(g);
    
    printf("Arithmetic EXEC nodes seeded. New edge count: %llu\n", 
           (unsigned long long)g->edge_count);
    
    melvin_close(g);
    return 0;
}

