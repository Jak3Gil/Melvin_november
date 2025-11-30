/*
 * mc_compile_agent.c - Compiler Agent (blob machine code)
 * 
 * This implements self-compilation inside the blob.
 * 
 * IMPORTANT: This is a TOOL file - compiled and embedded into .m blob.
 * It is NOT linked into the runtime loader.
 * 
 * The blob code calls this to compile C source and store machine code back into blob.
 */

#include "melvin.h"
#include <string.h>
#include <stdlib.h>

/* Helper: Ingest bytes as energy into graph */
static void ingest_bytes_as_energy(Graph *g, const uint8_t *bytes, size_t len, uint32_t port_node) {
    /* For each byte, bump activation of corresponding DATA node */
    for (size_t i = 0; i < len; i++) {
        uint8_t b = bytes[i];
        uint32_t data_id = (uint32_t)b;
        
        if (data_id < g->hdr->node_count) {
            /* Inject energy into data node */
            g->nodes[data_id].a += 0.1f;
            
            /* Create SEQ edge from previous byte (if exists) */
            if (i > 0) {
                uint8_t prev_b = bytes[i-1];
                uint32_t prev_id = (uint32_t)prev_b;
                
                if (prev_id < g->hdr->node_count) {
                    /* Find or create edge */
                    uint32_t eid = g->nodes[prev_id].first_out;
                    uint32_t found = UINT32_MAX;
                    while (eid != UINT32_MAX && eid < g->hdr->edge_count) {
                        if (g->edges[eid].dst == data_id) {
                            found = eid;
                            break;
                        }
                        eid = g->edges[eid].next_out;
                    }
                    
                    /* Create edge if not found */
                    if (found == UINT32_MAX && g->hdr->edge_count < (g->hdr->edges_offset - g->hdr->nodes_offset) / sizeof(Node)) {
                        uint32_t new_eid = (uint32_t)g->hdr->edge_count++;
                        g->edges[new_eid].src = prev_id;
                        g->edges[new_eid].dst = data_id;
                        g->edges[new_eid].w = 0.1f;
                        g->edges[new_eid].next_out = g->nodes[prev_id].first_out;
                        g->edges[new_eid].next_in = g->nodes[data_id].first_in;
                        g->nodes[prev_id].first_out = new_eid;
                        g->nodes[data_id].first_in = new_eid;
                        g->nodes[prev_id].out_degree++;
                        g->nodes[data_id].in_degree++;
                    }
                }
            }
        }
    }
}

/* Compiler agent - compiles C source and stores in blob */
void mc_compile_c(Graph *g, const char *src_path, uint64_t dest_blob_offset, uint64_t max_len) {
    if (!g || !g->hdr || !src_path) return;
    
    /* Get syscalls */
    MelvinSyscalls *sys = melvin_get_syscalls_from_blob(g);
    if (!sys || !sys->sys_run_cc || !sys->sys_read_file) return;
    
    /* Check blob bounds */
    if (dest_blob_offset >= g->hdr->blob_size) return;
    if (dest_blob_offset + max_len > g->hdr->blob_size) {
        max_len = g->hdr->blob_size - dest_blob_offset;
    }
    
    /* Step 1: Run compiler on source -> temp output */
    const char *out_path = "/tmp/melvin_cc_out.bin";
    if (sys->sys_run_cc(src_path, out_path) != 0) {
        return; /* Compile failed */
    }
    
    /* Step 2: Read compiled binary */
    uint8_t *bin = NULL;
    size_t bin_len = 0;
    if (sys->sys_read_file(out_path, &bin, &bin_len) != 0) {
        return;
    }
    
    if (bin_len > max_len) {
        bin_len = max_len; /* Truncate if necessary */
    }
    
    /* Step 3: Copy into blob region */
    uint8_t *blob_dest = g->blob + dest_blob_offset;
    for (size_t i = 0; i < bin_len; i++) {
        blob_dest[i] = bin[i];
    }
    
    /* Step 4: Ingest compiled bytes as energy (same as source bytes) */
    /* This makes machine code part of the same graph energy landscape */
    ingest_bytes_as_energy(g, bin, bin_len, 0); /* Use node 0 as port for now */
    
    /* Step 5: Also ingest source file bytes if we can read it */
    uint8_t *src_buf = NULL;
    size_t src_len = 0;
    if (sys->sys_read_file(src_path, &src_buf, &src_len) == 0) {
        ingest_bytes_as_energy(g, src_buf, src_len, 0);
        free(src_buf);
    }
    
    free(bin);
}

/* Helper: Find free blob region */
uint64_t mc_find_free_blob_region(Graph *g, uint64_t needed_size) {
    if (!g || !g->hdr) return 0;
    
    /* Simple linear search for free region */
    /* In production, blob code might maintain a free list */
    uint64_t search_start = 4096; /* Skip first 4KB (reserved for entrypoints) */
    uint64_t search_end = g->hdr->blob_size;
    
    for (uint64_t start = search_start; start + needed_size <= search_end; start += 1024) {
        /* Check if region is mostly zero (simple free check) */
        int is_free = 1;
        for (uint64_t i = 0; i < needed_size && i < 1024; i++) {
            if (g->blob[start + i] != 0) {
                is_free = 0;
                break;
            }
        }
        if (is_free) {
            return start;
        }
    }
    
    return 0; /* No free region found */
}

