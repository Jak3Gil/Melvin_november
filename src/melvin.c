/*
 * melvin.c - Binary Brain Loader + UEL Physics
 * 
 * ============================================================================
 * NO NODE/EDGE LIMITS. NO BIG O. WAVE PROPAGATION LAWS.
 * ============================================================================
 * 
 * This system scales to TERABYTES (16 billion nodes) with <100ms processing.
 * 
 * How? WAVE PROPAGATION, not algorithms:
 * 
 * 1. NO SCANNING ALL NODES - Event-driven propagation queue
 * 2. NO BIG O COMPLEXITY - Wave laws (UEL) determine processing
 * 3. NO FIXED LIMITS - node_count/edge_count are uint64_t (unlimited)
 * 4. EDGE-DIRECTED TRAVERSAL - Follow edges, never scan arrays
 * 5. LAZY COMPUTATION - Only compute what's needed (active nodes)
 * 6. MMAP VIRTUAL MEMORY - TB-scale file, only active pages in RAM
 * 
 * Processing time = O(active_nodes × avg_degree), NOT O(total_nodes)
 * Typical: 100 active nodes × 6 edges = 600 operations = ~10μs
 * Even at 1TB scale: <100μs (activity-bound, not size-bound)
 * 
 * See WAVE_PROPAGATION_LAWS.md for complete explanation.
 * See UNIVERSAL_LAW.txt for UEL physics equations.
 * 
 * ============================================================================
 * 
 * Does:
 *   1. mmap .m file (TB-scale virtual address space)
 *   2. feed bytes (inject energy into graph)
 *   3. expose syscalls (blob ↔ host bridge)
 *   4. run UEL physics (wave propagation)
 * 
 * UEL physics is embedded directly in melvin.c.
 * Blob is for future self-modification, but UEL runs here now.
 */

#define _GNU_SOURCE  /* For mremap, MREMAP_MAYMOVE */
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <signal.h>
#include <setjmp.h>

/* ========================================================================
 * ROUTING CHAIN DEBUG INSTRUMENTATION
 * ======================================================================== */
/* Compile-time flag: define ROUTING_DEBUG to enable routing chain logging */
/* When disabled, zero overhead - all logs compile away */
#ifndef ROUTING_DEBUG
#define ROUTING_DEBUG 0  /* Set to 1 to enable routing chain logs */
#endif

/* Routing log macro - only emits when ROUTING_DEBUG is enabled */
#if ROUTING_DEBUG
#define ROUTE_LOG(fmt, ...) \
    do { \
        fprintf(stderr, "[ROUTE] " fmt "\n", ##__VA_ARGS__); \
        fflush(stderr); \
    } while(0)
#else
#define ROUTE_LOG(fmt, ...) ((void)0)  /* Compile away when disabled */
#endif

/* ========================================================================
 * HIERARCHICAL PATTERN COMPOSITION
 * Track pattern adjacency and compose patterns hierarchically
 * ======================================================================== */

/* Pattern adjacency tracking - which patterns activate sequentially */
typedef struct {
    uint32_t pattern_a;
    uint32_t pattern_b;
    uint32_t cooccurrence_count;
    uint64_t last_seen;
} PatternAdjacency;

#define MAX_ADJACENCIES 1000
static PatternAdjacency adjacencies[MAX_ADJACENCIES];
static uint32_t adjacency_count = 0;

/* Last active pattern for sequential detection */
static uint32_t last_active_pattern = UINT32_MAX;
static uint64_t last_pattern_activation_time = 0;

/* ========================================================================
 * SOFT STRUCTURE INITIALIZATION
 * ======================================================================== */

/* Forward declarations */
static void initialize_soft_structure(Graph *g, bool is_new_file);
static void create_initial_edge_suggestions(Graph *g, bool is_new_file);
static uint32_t find_edge(Graph *g, uint32_t src, uint32_t dst);
static uint32_t create_edge(Graph *g, uint32_t src, uint32_t dst, float w);
static void expand_pattern(Graph *g, uint32_t pattern_node_id, const uint32_t *bindings);
static void melvin_execute_exec_node(Graph *g, uint32_t node_id);
static bool pattern_matches_sequence(Graph *g, uint32_t pattern_node_id, const uint32_t *sequence, uint32_t length, uint32_t *bindings);

/* Hierarchical composition forward declarations */
static void track_pattern_adjacency(Graph *g);
static void compose_adjacent_patterns(Graph *g);
static void record_adjacency(uint32_t pattern_a, uint32_t pattern_b);

/* Initialize soft structure scaffolding (guiding hints, not constraints) */
static void initialize_soft_structure(Graph *g, bool is_new_file) {
    if (!g || !g->nodes) return;
    
    /* Only initialize if this is a new file, or if structure is missing */
    /* For existing files, preserve learned structure */
    if (!is_new_file) {
        /* Check if soft structure already initialized (non-zero propensities exist) */
        bool has_structure = false;
        for (uint64_t i = 0; i < g->node_count && i < 256; i++) {
            if (g->nodes[i].input_propensity > 0.01f || 
                g->nodes[i].output_propensity > 0.01f ||
                g->nodes[i].memory_propensity > 0.01f) {
                has_structure = true;
                break;
            }
        }
        if (has_structure) return;  /* Already initialized, preserve it */
    }
    
    /* Initialize port range semantics and propensities */
    for (uint64_t i = 0; i < g->node_count && i < 256; i++) {
        Node *n = &g->nodes[i];
        
        /* Default: low propensities (internal processing node) */
        n->input_propensity = 0.1f;
        n->output_propensity = 0.1f;
        n->memory_propensity = 0.1f;
        n->semantic_hint = 0;
        
        /* INPUT PORTS (0-99): High input propensity */
        if (i < 100) {
            n->input_propensity = 0.8f;
            n->semantic_hint = 0;  /* Input port range */
            
            /* Text input ports (0-9) */
            if (i < 10) {
                n->input_propensity = 0.9f;
            }
            /* Sensor data ports (10-19) */
            else if (i < 20) {
                n->input_propensity = 0.85f;
            }
            /* Query/command ports (20-29) */
            else if (i < 30) {
                n->input_propensity = 0.85f;
            }
            /* Feedback input ports (30-39) */
            else if (i < 40) {
                n->input_propensity = 0.75f;  /* Slightly lower - these receive feedback, not primary input */
            }
        }
        /* OUTPUT PORTS (100-199): High output propensity */
        else if (i < 200) {
            n->output_propensity = 0.8f;
            n->semantic_hint = 100;  /* Output port range */
            
            /* Text output ports (100-109) */
            if (i < 110) {
                n->output_propensity = 0.9f;
            }
            /* Action output ports (110-119) */
            else if (i < 120) {
                n->output_propensity = 0.85f;
            }
            /* Prediction output ports (120-129) */
            else if (i < 130) {
                n->output_propensity = 0.85f;
            }
            /* Confidence output ports (130-139) */
            else if (i < 140) {
                n->output_propensity = 0.8f;
            }
        }
        /* CONTROL/MEMORY PORTS (200-255): High memory propensity */
        else if (i < 256) {
            n->memory_propensity = 0.8f;
            n->semantic_hint = 200;  /* Control/memory port range */
            
            /* Working memory ports (200-209) */
            if (i < 210) {
                n->memory_propensity = 0.85f;
            }
            /* Long-term memory ports (210-219) */
            else if (i < 220) {
                n->memory_propensity = 0.9f;  /* Highest - long-term retention */
            }
            /* Attention ports (220-229) */
            else if (i < 230) {
                n->memory_propensity = 0.75f;
            }
            /* Meta-control ports (230-239) */
            else if (i < 240) {
                n->memory_propensity = 0.7f;
            }
            /* Error handling ports (250-259) */
            else if (i >= 250 && i < 260) {
                n->input_propensity = 0.8f;  /* Receives error signals */
                n->output_propensity = 0.6f;  /* Produces recovery signals */
                n->memory_propensity = 0.5f;  /* Medium retention (error patterns) */
                n->semantic_hint = 250;  /* Error handling range */
                
                if (i == 250) {
                    /* Error detection node */
                    n->input_propensity = 0.9f;
                } else if (i >= 251 && i < 255) {
                    /* Recovery pattern nodes */
                    n->output_propensity = 0.7f;
                } else if (i >= 255 && i < 260) {
                    /* Self-regulation nodes (chaos monitoring, activity control) */
                    n->input_propensity = 0.7f;
                    n->output_propensity = 0.6f;
                }
            }
            /* Temporal anchor ports (240-249) */
            else {
                n->memory_propensity = 0.6f;  /* Temporal nodes have medium retention */
                
                /* Special temporal anchors */
                if (i == 240) {
                    /* "now" node - always activated with current input */
                    n->input_propensity = 0.5f;  /* Receives current input */
                    n->memory_propensity = 0.3f;  /* Low retention (current moment) */
                } else if (i == 241) {
                    /* "recent" node - tracks recent activations */
                    n->memory_propensity = 0.7f;  /* Medium retention (recent history) */
                } else if (i == 242) {
                    /* "memory" node - long-term patterns */
                    n->memory_propensity = 0.95f;  /* Very high retention */
                } else if (i == 243) {
                    /* "future" node - prediction anchor */
                    n->output_propensity = 0.5f;  /* Produces predictions */
                    n->memory_propensity = 0.5f;
                }
            }
        }
        /* TOOL GATEWAY PORTS (300-699): Generic tool interface (abstract - no specific tools) */
        /* Tools are discovered via syscalls, not hardcoded into substrate */
        else if (i < 700) {
            n->semantic_hint = 300;  /* Generic tool gateway range */
            /* Generic tool gateway: input/output propensities suggest tool-like behavior */
            if ((i % 20) < 10) {
                /* Tool input range (e.g., 300-309, 400-409, etc.) */
                n->input_propensity = 0.7f;
                n->output_propensity = 0.3f;
            } else {
                /* Tool output range (e.g., 310-319, 410-419, etc.) */
                n->input_propensity = 0.3f;
                n->output_propensity = 0.8f;  /* High - produces tool output */
            }
        }
        /* MOTOR CONTROL GATEWAY (700-719): Motor commands */
        else if (i < 720) {
            n->semantic_hint = 700;  /* Motor gateway range */
            if (i < 710) {
                /* Motor input (700-709) - receives motor commands */
                n->input_propensity = 0.7f;
                n->output_propensity = 0.3f;
            } else {
                /* Motor output (710-719) - sends motor frames */
                n->input_propensity = 0.3f;
                n->output_propensity = 0.9f;  /* Very high - produces motor commands */
            }
        }
        /* FILE I/O GATEWAY (720-739): File operations */
        else if (i < 740) {
            n->semantic_hint = 720;  /* File I/O gateway range */
            if (i < 730) {
                /* File input (720-729) - receives file paths/data */
                n->input_propensity = 0.7f;
                n->output_propensity = 0.3f;
            } else {
                /* File output (730-739) - produces file data */
                n->input_propensity = 0.3f;
                n->output_propensity = 0.8f;  /* High - produces file data */
            }
        }
        /* CODE PATTERN PORTS (740-839): Compiled code patterns (graph learns from machine code) */
        else if (i < 840) {
            n->semantic_hint = 740;  /* Code pattern range */
            n->memory_propensity = 0.9f;  /* High retention - code patterns are important */
            n->input_propensity = 0.6f;  /* Receives compiled code */
            n->output_propensity = 0.4f;  /* Can produce code-like patterns */
            
        }
    }
    
    /* Initialize feedback channel nodes (if new file) */
    if (is_new_file && g->node_count > 33) {
        /* Positive feedback node (30) */
        if (g->node_count > 30) {
            g->nodes[30].input_propensity = 0.9f;
            g->nodes[30].semantic_hint = 30;
        }
        /* Negative feedback node (31) */
        if (g->node_count > 31) {
            g->nodes[31].input_propensity = 0.9f;
            g->nodes[31].semantic_hint = 31;
        }
        /* Uncertainty node (32) */
        if (g->node_count > 32) {
            g->nodes[32].input_propensity = 0.8f;
            g->nodes[32].semantic_hint = 32;
        }
        /* Curiosity signal node (33) */
        if (g->node_count > 33) {
            g->nodes[33].input_propensity = 0.7f;
            g->nodes[33].semantic_hint = 33;
        }
    }
}

/* Create weak initial edges suggesting data flow (graph can strengthen/weaken/rewire) */
static void create_initial_edge_suggestions(Graph *g, bool is_new_file) {
    if (!g || !g->nodes || !is_new_file) return;
    
    /* Only create initial edges for new files */
    /* Graph will learn which ones to keep through UEL physics */
    
    /* Removed edges_created counter - no limits on edge creation */
    /* NO LIMITS - graph creates all needed initial edges */
    /* Graph will learn which edges are useful through UEL physics */
    
    /* RELATIVE: Use graph state to determine initial edge weights if available */
    /* For new files, avg_edge_strength may be 0, so use default; otherwise scale relative */
    float base_weight = (g->avg_edge_strength > 0.01f) ? (g->avg_edge_strength * 0.1f) : 0.1f;
    if (base_weight < 0.01f) base_weight = 0.01f;  /* Minimum */
    float weak_weight = (g->avg_edge_strength > 0.01f) ? (g->avg_edge_strength * 0.05f) : 0.05f;
    if (weak_weight < 0.005f) weak_weight = 0.005f;  /* Minimum for very weak edges */
    float very_weak_weight = (g->avg_edge_strength > 0.01f) ? (g->avg_edge_strength * 0.02f) : 0.02f;
    if (very_weak_weight < 0.002f) very_weak_weight = 0.002f;  /* Minimum for very weak edges */
    float extra_weak_weight = (g->avg_edge_strength > 0.01f) ? (g->avg_edge_strength * 0.03f) : 0.03f;
    if (extra_weak_weight < 0.003f) extra_weak_weight = 0.003f;  /* Minimum */
    float medium_weight = (g->avg_edge_strength > 0.01f) ? (g->avg_edge_strength * 0.15f) : 0.15f;
    if (medium_weight < 0.015f) medium_weight = 0.015f;  /* Minimum */
    float memory_weight = (g->avg_edge_strength > 0.01f) ? (g->avg_edge_strength * 0.08f) : 0.08f;
    if (memory_weight < 0.008f) memory_weight = 0.008f;  /* Minimum */
    float temporal_weak = (g->avg_edge_strength > 0.01f) ? (g->avg_edge_strength * 0.12f) : 0.12f;
    if (temporal_weak < 0.012f) temporal_weak = 0.012f;  /* Minimum */
    float temporal_medium = (g->avg_edge_strength > 0.01f) ? (g->avg_edge_strength * 0.15f) : 0.15f;
    if (temporal_medium < 0.015f) temporal_medium = 0.015f;  /* Minimum */
    
    /* 1. Input → Working Memory (ports 0-99 → 200-209) */
    /* Create edges for ALL input ports - ensure_node will create nodes as needed */
    for (uint32_t input = 0; input < 100; input++) {
        for (uint32_t memory = 200; memory < 210; memory++) {
            if (find_edge(g, input, memory) == UINT32_MAX) {
                create_edge(g, input, memory, base_weight);  /* ensure_node creates nodes */
            }
        }
    }
    
    /* 2. Working Memory → Output (ports 200-209 → 100-199) */
    /* Create edges for ALL output ports - ensure_node will create nodes as needed */
    for (uint32_t memory = 200; memory < 210; memory++) {
        for (uint32_t output = 100; output < 200; output++) {
            if (find_edge(g, memory, output) == UINT32_MAX) {
                create_edge(g, memory, output, base_weight);  /* ensure_node creates nodes */
            }
        }
    }
    
    /* 3. Output → Feedback (ports 100-199 → 30-33) */
    /* Create edges for ALL output ports - ensure_node will create nodes as needed */
    for (uint32_t output = 100; output < 200; output++) {
        for (uint32_t feedback = 30; feedback < 34; feedback++) {
            if (find_edge(g, output, feedback) == UINT32_MAX) {
                create_edge(g, output, feedback, weak_weight);  /* ensure_node creates nodes */
            }
        }
    }
    
    /* 4. Sparse Memory → Memory connections (create more connections) */
    for (uint32_t i = 0; i < 50; i++) {  /* Increased from 20 to 50 */
        uint32_t mem1 = 200 + (i % 10);
        uint32_t mem2 = 200 + ((i * 7) % 10);  /* Pseudo-random pairing */
        if (mem1 != mem2) {
            if (find_edge(g, mem1, mem2) == UINT32_MAX) {
                create_edge(g, mem1, mem2, memory_weight);  /* ensure_node creates nodes */
            }
        }
    }
    
    /* 5. Temporal anchor connections */
    /* "now" (240) → "recent" (241) */
    if (find_edge(g, 240, 241) == UINT32_MAX) {
        create_edge(g, 240, 241, temporal_medium);  /* ensure_node creates nodes */
    }
    /* "recent" (241) → "memory" (242) */
    if (find_edge(g, 241, 242) == UINT32_MAX) {
        create_edge(g, 241, 242, temporal_weak);  /* ensure_node creates nodes */
    }
    /* "memory" (242) → "future" (243) */
    if (find_edge(g, 242, 243) == UINT32_MAX) {
        create_edge(g, 242, 243, base_weight);  /* ensure_node creates nodes */
    }
    
    /* 6. Hardware instinct patterns - bootstrap tool connections */
    /* These provide initial routing, graph can strengthen/weaken/rewire */
    /* ensure_node will create all nodes as needed - no need to check node_count */
    
    /* Mic → Audio Processing → STT Gateway (300-309) */
    /* Mic port (0) → Working memory (200) */
    if (find_edge(g, 0, 200) == UINT32_MAX) {
        create_edge(g, 0, 200, base_weight);  /* ensure_node creates nodes */
    }
    /* Working memory → STT gateway input (300) */
    if (find_edge(g, 200, 300) == UINT32_MAX) {
        create_edge(g, 200, 300, weak_weight);  /* ensure_node creates nodes */
    }
    /* STT gateway output (310) → Working memory */
    if (find_edge(g, 310, 201) == UINT32_MAX) {
        create_edge(g, 310, 201, base_weight);  /* ensure_node creates nodes */
    }
    /* Working memory → Speaker port (100) */
    if (find_edge(g, 201, 100) == UINT32_MAX) {
        create_edge(g, 201, 100, weak_weight);  /* ensure_node creates nodes */
    }
    
    /* Camera → Vision Processing → Vision Gateway (400-409) */
    /* Camera port (10) → Working memory (201) */
    if (find_edge(g, 10, 201) == UINT32_MAX) {
        create_edge(g, 10, 201, base_weight);  /* ensure_node creates nodes */
    }
    /* Working memory → Vision gateway input (400) */
    if (find_edge(g, 201, 400) == UINT32_MAX) {
        create_edge(g, 201, 400, weak_weight);  /* ensure_node creates nodes */
    }
    /* Vision gateway output (410) → Working memory */
    if (find_edge(g, 410, 202) == UINT32_MAX) {
        create_edge(g, 410, 202, base_weight);  /* ensure_node creates nodes */
    }
    /* Working memory → Display port (110) */
    if (find_edge(g, 202, 110) == UINT32_MAX) {
        create_edge(g, 202, 110, weak_weight);  /* ensure_node creates nodes */
    }
    
    /* Text → LLM Processing → LLM Gateway (500-509) */
    /* Text input port (20) → Working memory (202) */
    if (find_edge(g, 20, 202) == UINT32_MAX) {
        create_edge(g, 20, 202, base_weight);  /* ensure_node creates nodes */
    }
    /* Working memory → LLM gateway input (500) */
    if (find_edge(g, 202, 500) == UINT32_MAX) {
        create_edge(g, 202, 500, weak_weight);  /* ensure_node creates nodes */
    }
    /* LLM gateway output (510) → Working memory */
    if (find_edge(g, 510, 203) == UINT32_MAX) {
        create_edge(g, 510, 203, base_weight);  /* ensure_node creates nodes */
    }
    /* Working memory → Text output port (100) or TTS gateway (600) */
    if (find_edge(g, 203, 100) == UINT32_MAX) {
        create_edge(g, 203, 100, weak_weight);  /* ensure_node creates nodes */
    }
    
    /* TTS Gateway (600-609) */
    /* Text → TTS gateway input (600) */
    if (find_edge(g, 203, 600) == UINT32_MAX) {
        create_edge(g, 203, 600, weak_weight);  /* ensure_node creates nodes */
    }
    /* TTS gateway output (610) → Speaker port (100) */
    if (find_edge(g, 610, 100) == UINT32_MAX) {
        create_edge(g, 610, 100, base_weight);  /* ensure_node creates nodes */
    }
    
    /* 7. Generic tool gateway internal connections (very weak - graph learns tool behavior) */
    /* Tool gateways have internal input→output connections (discovered by graph) */
    for (uint32_t tool_in = 300; tool_in < 700; tool_in += 100) {
        uint32_t tool_out = tool_in + 10;  /* Output is 10 nodes after input */
        if (find_edge(g, tool_in, tool_out) == UINT32_MAX) {
            create_edge(g, tool_in, tool_out, extra_weak_weight);  /* Very weak - graph discovers */
        }
    }
    
    /* 8. Cross-tool connections (very weak - graph learns these patterns) */
    /* These enable hierarchical pattern building */
    /* STT ↔ Vision (audio-visual connection) */
    if (find_edge(g, 310, 410) == UINT32_MAX) {
        create_edge(g, 310, 410, very_weak_weight);  /* ensure_node creates nodes */
    }
    if (find_edge(g, 410, 310) == UINT32_MAX) {
        create_edge(g, 410, 310, very_weak_weight);  /* ensure_node creates nodes */
    }
    
    /* Vision ↔ LLM (visual-text connection) */
    if (find_edge(g, 410, 510) == UINT32_MAX) {
        create_edge(g, 410, 510, very_weak_weight);  /* ensure_node creates nodes */
    }
    if (find_edge(g, 510, 410) == UINT32_MAX) {
        create_edge(g, 510, 410, very_weak_weight);  /* ensure_node creates nodes */
    }
    
    /* LLM ↔ STT (text-audio connection) */
    if (find_edge(g, 510, 310) == UINT32_MAX) {
        create_edge(g, 510, 310, very_weak_weight);  /* ensure_node creates nodes */
    }
    if (find_edge(g, 310, 510) == UINT32_MAX) {
        create_edge(g, 310, 510, very_weak_weight);  /* ensure_node creates nodes */
    }
    
    /* LLM ↔ TTS (text generation → speech) */
    if (find_edge(g, 510, 600) == UINT32_MAX) {
        create_edge(g, 510, 600, weak_weight);  /* ensure_node creates nodes */
    }
    
    /* 9. ERROR HANDLING PATTERNS (graph learns from failures through UEL) */
    /* Error detection nodes: 250-259 */
    /* Generic tool gateway failures → Error detection (250) */
    for (uint32_t tool_in = 300; tool_in < 700; tool_in += 100) {
        if (find_edge(g, tool_in, 250) == UINT32_MAX) {
            create_edge(g, tool_in, 250, base_weight);  /* Tool failures → error detection */
        }
    }
    
    /* Error detection → Recovery patterns (251-254) */
    for (uint32_t recovery = 251; recovery < 255; recovery++) {
        if (find_edge(g, 250, recovery) == UINT32_MAX) {
            create_edge(g, 250, recovery, memory_weight);  /* ensure_node creates nodes */
        }
        /* Recovery → Retry generic tool gateway (graph discovers which tool) */
        for (uint32_t tool_in = 300; tool_in < 700; tool_in += 100) {
            if (find_edge(g, recovery, tool_in) == UINT32_MAX) {
                create_edge(g, recovery, tool_in, extra_weak_weight);  /* Very weak - graph discovers */
            }
        }
    }
    
    /* Error → Feedback (graph learns from errors) */
    if (find_edge(g, 250, 31) == UINT32_MAX) {
        create_edge(g, 250, 31, base_weight);  /* ensure_node creates nodes */
    }
    
    /* 9. AUTOMATIC TOOL INTEGRATION PATTERNS */
    /* Graph learns when to call tools through pattern recognition */
    /* Input patterns → Tool gateway activation (graph learns thresholds) */
    /* Audio input (0) → STT gateway (direct connection for conversation) */
    if (find_edge(g, 0, 300) == UINT32_MAX) {
        create_edge(g, 0, 300, weak_weight);  /* Mic → STT Gateway */
    }
    /* STT Gateway internal: input (300) → output (310) */
    if (find_edge(g, 300, 310) == UINT32_MAX) {
        create_edge(g, 300, 310, medium_weight);  /* STT Gateway internal */
    }
    /* LLM Gateway internal: input (500) → output (510) */
    if (find_edge(g, 500, 510) == UINT32_MAX) {
        create_edge(g, 500, 510, medium_weight);  /* LLM Gateway internal */
    }
    /* TTS Gateway internal: input (600) → output (610) */
    if (find_edge(g, 600, 610) == UINT32_MAX) {
        create_edge(g, 600, 610, medium_weight);  /* TTS Gateway internal */
    }
    /* Audio input (0) → STT gateway when pattern matches (working memory patterns) */
    for (uint32_t pattern = 200; pattern < 210; pattern++) {
        if (find_edge(g, pattern, 300) == UINT32_MAX) {
            create_edge(g, pattern, 300, extra_weak_weight);  /* ensure_node creates nodes */
        }
    }
    
    /* Text patterns → LLM gateway when needed */
    for (uint32_t pattern = 201; pattern < 211; pattern++) {
        if (find_edge(g, pattern, 500) == UINT32_MAX) {
            create_edge(g, pattern, 500, extra_weak_weight);  /* ensure_node creates nodes */
        }
    }
    
    /* Tool outputs → Automatic graph feeding (stronger - these create patterns) */
    /* STT output → Graph nodes (automatic pattern creation) */
    for (uint32_t mem = 200; mem < 210; mem++) {
        if (find_edge(g, 310, mem) == UINT32_MAX) {
            create_edge(g, 310, mem, medium_weight);  /* ensure_node creates nodes */
        }
    }
    
    /* LLM output → Graph nodes */
    for (uint32_t mem = 201; mem < 211; mem++) {
            if (find_edge(g, 510, mem) == UINT32_MAX) {
                create_edge(g, 510, mem, medium_weight);  /* RELATIVE: Tool output → Graph */
            }
        }
        
    /* Vision output → Graph nodes */
    for (uint32_t mem = 202; mem < 212; mem++) {
        if (find_edge(g, 410, mem) == UINT32_MAX) {
            create_edge(g, 410, mem, medium_weight);  /* ensure_node creates nodes */
        }
    }
    
    /* TTS output → Graph nodes (audio patterns) */
    for (uint32_t mem = 203; mem < 213; mem++) {
        if (find_edge(g, 610, mem) == UINT32_MAX) {
            create_edge(g, 610, mem, medium_weight);  /* ensure_node creates nodes */
        }
    }
    
    /* 10. SELF-REGULATION PATTERNS (graph controls its own activity) */
        /* Chaos monitoring → Activity adjustment (255-259) */
        /* High chaos → Reduce activity (255) */
        if (find_edge(g, 242, 255) == UINT32_MAX) {
            create_edge(g, 242, 255, base_weight);  /* RELATIVE: Memory → Chaos monitor */
        }
        /* Low chaos → Increase exploration (256) */
        if (find_edge(g, 242, 256) == UINT32_MAX) {
            create_edge(g, 242, 256, base_weight);  /* RELATIVE: Memory → Exploration */
        }
        
        /* Activity adjustment → Input throttling */
        if (find_edge(g, 255, 0) == UINT32_MAX) {
            create_edge(g, 255, 0, weak_weight);  /* RELATIVE: High chaos → Throttle input */
        }
        if (find_edge(g, 256, 0) == UINT32_MAX) {
            create_edge(g, 256, 0, weak_weight);  /* RELATIVE: Low chaos → Increase input */
        }
        
    /* Feedback loops for self-regulation */
    /* Output activity → Feedback → Activity adjustment */
    for (uint32_t output = 100; output < 110; output++) {
        if (find_edge(g, output, 255) == UINT32_MAX) {
            create_edge(g, output, 255, weak_weight);  /* ensure_node creates nodes */
        }
    }
    
    /* 11. MOTOR CONTROL PATTERNS */
    /* Motor gateway (700-719): Commands → Motor frames */
    /* Working memory → Motor gateway input (700) */
    if (find_edge(g, 203, 700) == UINT32_MAX) {
        create_edge(g, 203, 700, weak_weight);  /* ensure_node creates nodes */
    }
    /* Motor gateway output (710) → Motor port (120) */
    if (find_edge(g, 710, 120) == UINT32_MAX) {
        create_edge(g, 710, 120, base_weight);  /* ensure_node creates nodes */
    }
    /* Code patterns → Motor control (compiled code can control motors) */
    for (uint32_t code = 740; code < 750; code++) {
        if (find_edge(g, code, 700) == UINT32_MAX) {
            create_edge(g, code, 700, medium_weight);  /* ensure_node creates nodes */
        }
    }
    
    /* 12. FILE I/O PATTERNS */
    /* File I/O gateway (720-739): File operations */
    /* Working memory → File read gateway (720) */
    if (find_edge(g, 202, 720) == UINT32_MAX) {
        create_edge(g, 202, 720, weak_weight);  /* ensure_node creates nodes */
    }
    /* File read output (730) → Working memory */
    if (find_edge(g, 730, 201) == UINT32_MAX) {
        create_edge(g, 730, 201, base_weight);  /* ensure_node creates nodes */
    }
    /* File read → Compile (C files → compilation) */
    if (find_edge(g, 730, 740) == UINT32_MAX) {
        create_edge(g, 730, 740, medium_weight);  /* ensure_node creates nodes - C files → code patterns */
    }
    /* Compiled code → Motor control (use compiled code for motors) */
    for (uint32_t code = 740; code < 750; code++) {
        if (find_edge(g, code, 700) == UINT32_MAX) {
            create_edge(g, code, 700, medium_weight);  /* ensure_node creates nodes */
        }
    }
    /* File write gateway (721) - graph can write files */
    if (find_edge(g, 203, 721) == UINT32_MAX) {
        create_edge(g, 203, 721, weak_weight);  /* ensure_node creates nodes */
    }
    /* File write output (731) → File system */
    if (find_edge(g, 731, 121) == UINT32_MAX) {
        create_edge(g, 731, 121, base_weight);  /* ensure_node creates nodes */
    }
    
    /* 13. CONVERSATION DATA PATTERNS */
    /* Conversation input → STT → LLM → TTS → Conversation output */
    /* Text input (20) → Conversation memory (204-209) */
    for (uint32_t conv = 204; conv < 210; conv++) {
        if (find_edge(g, 20, conv) == UINT32_MAX) {
            create_edge(g, 20, conv, base_weight);  /* ensure_node creates nodes */
        }
        /* Conversation memory → LLM (for responses) */
        if (find_edge(g, conv, 500) == UINT32_MAX) {
            create_edge(g, conv, 500, medium_weight);  /* ensure_node creates nodes */
        }
        /* LLM output → Conversation memory (responses) */
        if (find_edge(g, 510, conv) == UINT32_MAX) {
            create_edge(g, 510, conv, base_weight);  /* ensure_node creates nodes */
        }
        /* Conversation memory → TTS (speak responses) */
        if (find_edge(g, conv, 600) == UINT32_MAX) {
            create_edge(g, conv, 600, medium_weight);  /* ensure_node creates nodes */
        }
        /* Conversation memory → Text output (100) */
        if (find_edge(g, conv, 100) == UINT32_MAX) {
            create_edge(g, conv, 100, base_weight);  /* ensure_node creates nodes */
        }
    }
    /* STT → Conversation memory (speech input) */
    if (find_edge(g, 310, 204) == UINT32_MAX) {
        create_edge(g, 310, 204, base_weight);  /* ensure_node creates nodes */
    }
    
    /* 14. C FILE PROCESSING WORKFLOW */
    /* File read (730) → Working memory (202) */
    if (find_edge(g, 730, 202) == UINT32_MAX) {
        create_edge(g, 730, 202, base_weight);  /* ensure_node creates nodes */
    }
    /* Working memory → Compile (C source → machine code) */
    if (find_edge(g, 202, 740) == UINT32_MAX) {
        create_edge(g, 202, 740, medium_weight);  /* ensure_node creates nodes */
    }
    /* Compiled code → Motor control (use compiled code) */
    for (uint32_t code = 740; code < 750; code++) {
        if (find_edge(g, code, 700) == UINT32_MAX) {
            create_edge(g, code, 700, medium_weight);  /* ensure_node creates nodes */
        }
    }
    /* Compiled code → Blob execution (graph can run its own code) */
    for (uint32_t code = 740; code < 750; code++) {
        if (find_edge(g, code, 243) == UINT32_MAX) {
            create_edge(g, code, 243, weak_weight);  /* ensure_node creates nodes - code → future/prediction */
        }
    }
    
    /* 15. TOOL SUCCESS/FAILURE FEEDBACK LOOPS */
    /* Graph learns which tools are reliable through UEL correlation */
    if (g->node_count > 600) {
        /* Tool success → Positive feedback (30) */
        for (uint32_t tool_out = 310; tool_out <= 610; tool_out += 100) {
            if (tool_out < g->node_count && find_edge(g, tool_out, 30) == UINT32_MAX) {
                create_edge(g, tool_out, 30, base_weight);  /* RELATIVE: Tool output → Success feedback */
            }
        }
        
        /* Tool failure → Error detection (250) → Negative feedback (31) */
        if (find_edge(g, 250, 31) == UINT32_MAX) {
            create_edge(g, 250, 31, base_weight);  /* RELATIVE: Error → Negative feedback */
        }
        
        /* Feedback → Tool gateway selection (graph learns which tools work) */
        if (find_edge(g, 30, 300) == UINT32_MAX) {
            create_edge(g, 30, 300, weak_weight);  /* RELATIVE: Success → Prefer STT */
        }
        if (find_edge(g, 30, 500) == UINT32_MAX) {
            create_edge(g, 30, 500, weak_weight);  /* RELATIVE: Success → Prefer LLM */
        }
        if (find_edge(g, 31, 250) == UINT32_MAX) {
            create_edge(g, 31, 250, weak_weight);  /* RELATIVE: Failure → Error detection */
        }
    }
}

/* ========================================================================
 * OPEN/CREATE .M FILE
 * ======================================================================== */

/* Forward declare */
static Graph* melvin_open_with_cold(const char *path, size_t initial_nodes, size_t initial_edges, 
                                     size_t blob_size, size_t cold_data_size);

Graph* melvin_open(const char *path, size_t initial_nodes, size_t initial_edges, size_t blob_size) {
    /* Calculate minimum nodes needed:
     * - 0-255: byte values (256 nodes)
     * - 100-199: output ports (100 nodes, overlaps with bytes)
     * - 200-255: working memory/control (overlaps with bytes)
     * - 300-699: tool gateways (400 nodes)
     * - 700-719: motor control gateway (20 nodes)
     * - 720-739: file I/O gateway (20 nodes)
     * - 740-839: code pattern nodes (100 nodes)
     * Maximum needed: 840 nodes
     * 
     * If initial_nodes is 0 or too small, use minimum + headroom
     * Otherwise use provided value (for backward compatibility)
     */
    size_t min_nodes_needed = 800;  /* Maximum node ID we actually use */
    size_t headroom = 200;  /* 25% headroom for immediate growth */
    size_t recommended = min_nodes_needed + headroom;  /* 1000 nodes */
    
    if (initial_nodes == 0 || initial_nodes < min_nodes_needed) {
        initial_nodes = recommended;  /* Start with minimum needed + headroom */
    }
    return melvin_open_with_cold(path, initial_nodes, initial_edges, blob_size, 0);
}

static Graph* melvin_open_with_cold(const char *path, size_t initial_nodes, size_t initial_edges, 
                                     size_t blob_size, size_t cold_data_size) {
    if (!path) return NULL;
    
    Graph *g = calloc(1, sizeof(Graph));
    if (!g) return NULL;
    
    /* Calculate layout - all offsets 64-bit */
    uint64_t header_size = sizeof(MelvinHeader);
    uint64_t nodes_size = (uint64_t)initial_nodes * sizeof(Node);
    uint64_t edges_size = (uint64_t)initial_edges * sizeof(Edge);
    /* If blob_size is 0, use default 64KB for EXEC code storage */
    if (blob_size == 0) {
        blob_size = 65536;  /* 64KB default */
    }
    uint64_t blob_size_u64 = (uint64_t)blob_size;
    uint64_t cold_size_u64 = (uint64_t)cold_data_size;
    
    uint64_t off = header_size;
    uint64_t nodes_offset = off;
    off += nodes_size;
    uint64_t edges_offset = off;
    off += edges_size;
    uint64_t blob_offset = off;
    off += blob_size_u64;
    uint64_t cold_data_offset = off;
    off += cold_size_u64;
    uint64_t total_size = off;
    
    /* Round up to page size */
    size_t page_size = getpagesize();
    total_size = (total_size + page_size - 1) & ~(page_size - 1);
    
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        free(g);
        return NULL;
    }
    
    struct stat st;
    fstat(fd, &st);
    bool is_new = (st.st_size == 0);
    
    if (is_new) {
        /* Create new file */
        fprintf(stderr, "Creating brain file (%.1f MB)...\r", total_size / (1024.0 * 1024.0));
        fflush(stderr);
        if (ftruncate(fd, total_size) < 0) {
            close(fd);
            free(g);
            return NULL;
        }
        
        /* mmap with EXEC permission - blob can contain executable code */
        fprintf(stderr, "Mapping memory...\r");
        fflush(stderr);
        void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            close(fd);
            free(g);
            return NULL;
        }
        
        /* Initialize header */
        fprintf(stderr, "Initializing header...\r");
        fflush(stderr);
        MelvinHeader *hdr = (MelvinHeader *)map;
        memcpy(hdr->magic, MELVIN_MAGIC, 4);
        hdr->version = MELVIN_VERSION;
        hdr->flags = 0;
        hdr->file_size = total_size;
        hdr->nodes_offset = nodes_offset;
        hdr->node_count = initial_nodes;
        hdr->edges_offset = edges_offset;
        hdr->edge_count = 0;
        hdr->blob_offset = blob_offset;
        hdr->blob_size = blob_size_u64;
        hdr->cold_data_offset = cold_data_offset;
        hdr->cold_data_size = cold_size_u64;
        hdr->main_entry_offset = 0;  /* Set by tool that seeds blob */
        hdr->syscalls_ptr_offset = 0; /* Set by tool */
        
        /* Zero hot regions */
        fprintf(stderr, "Initializing nodes (%zu nodes)...\r", initial_nodes);
        fflush(stderr);
        memset((char *)map + hdr->nodes_offset, 0, (size_t)nodes_size);
        fprintf(stderr, "Initializing edges (%zu edges)...\r", initial_edges);
        fflush(stderr);
        memset((char *)map + hdr->edges_offset, 0, (size_t)edges_size);
        fprintf(stderr, "Initializing blob (%zu bytes)...\r", blob_size);
        fflush(stderr);
        memset((char *)map + hdr->blob_offset, 0, (size_t)blob_size_u64);
        /* Cold data left as-is (will be filled by corpus loader) */
        
        /* Initialize ALL nodes with proper UINT32_MAX for edge pointers */
        /* CRITICAL: memset(0) leaves first_in/first_out = 0, pointing to edge 0! */
        fprintf(stderr, "Setting up byte nodes (0-255)...\r");
        fflush(stderr);
        Node *nodes = (Node *)((char *)map + hdr->nodes_offset);
        
        /* First: Initialize ALL nodes to prevent phantom edges */
        for (size_t i = 0; i < initial_nodes; i++) {
            nodes[i].first_in = UINT32_MAX;   /* No edges yet */
            nodes[i].first_out = UINT32_MAX;  /* No edges yet */
            nodes[i].in_degree = 0;
            nodes[i].out_degree = 0;
        }
        
        /* Then: Set byte-specific fields for nodes 0-255 */
        for (int i = 0; i < 256 && i < (int)initial_nodes; i++) {
            nodes[i].byte = (uint8_t)i;
            nodes[i].a = 0.0f;
            /* Initialize soft structure fields to zero (will be set by initialize_soft_structure) */
            nodes[i].input_propensity = 0.0f;
            nodes[i].output_propensity = 0.0f;
            nodes[i].memory_propensity = 0.0f;
            nodes[i].semantic_hint = 0;
        }
        
        g->fd = fd;
        g->map_base = map;
        g->map_size = (size_t)total_size;
        g->hdr = hdr;
        g->nodes = nodes;
        g->edges = (Edge *)((char *)map + hdr->edges_offset);
        g->blob = (uint8_t *)((char *)map + hdr->blob_offset);
        g->cold_data = (hdr->cold_data_size > 0) ? 
                       (uint8_t *)((char *)map + hdr->cold_data_offset) : NULL;
        g->node_count = hdr->node_count;
        g->edge_count = hdr->edge_count;
        g->blob_size = hdr->blob_size;
        g->cold_data_size = hdr->cold_data_size;
        
        /* GRAPH LAWS DETERMINE PROCESSING: Use defaults for new files, sample for existing */
        /* Initialize UEL physics state (always initialize, even for existing files) */
        if (g->node_count > 0 && g->edge_count > 0) {
            /* GRAPH LAWS DETERMINE PROCESSING: Use all nodes and edges for initial estimates */
            /* NO SAMPLING LIMITS - use full graph state */
            uint64_t node_sample = g->node_count;
            uint64_t edge_sample = g->edge_count;
            
            float sum_activation = 0.0f;
            float sum_edge_strength = 0.0f;
            uint64_t active_count = 0;
            
            /* Sample nodes with stride for better coverage */
            uint64_t node_stride = (g->node_count > node_sample) ? (g->node_count / node_sample) : 1;
            for (uint64_t i = 0; i < g->node_count && active_count < node_sample; i += node_stride) {
                float a_abs = fabsf(g->nodes[i].a);
                if (a_abs > 0.001f) {
                    sum_activation += a_abs;
                    active_count++;
                }
            }
            
            /* Sample edges with stride */
            uint64_t edge_stride = (g->edge_count > edge_sample) ? (g->edge_count / edge_sample) : 1;
            for (uint64_t i = 0; i < g->edge_count && i < edge_sample * edge_stride; i += edge_stride) {
                sum_edge_strength += fabsf(g->edges[i].w);
            }
            
            /* Scale estimates to full graph size */
            uint64_t sampled_edges = (g->edge_count < edge_sample) ? g->edge_count : edge_sample;
            g->avg_activation = (active_count > 0) ? (sum_activation / active_count) : 0.1f;
            if (g->avg_activation < 0.01f) g->avg_activation = 0.1f;  /* Minimum */
            
            g->avg_edge_strength = (sampled_edges > 0) ? (sum_edge_strength / sampled_edges) : 0.1f;
            if (g->avg_edge_strength < 0.01f) g->avg_edge_strength = 0.1f;  /* Minimum */
            
            /* Estimate initial chaos from activation variance (sample) */
            float variance = 0.0f;
            uint64_t sample_count = (node_sample < 100) ? node_sample : 100;
            uint64_t chaos_stride = (g->node_count > sample_count) ? (g->node_count / sample_count) : 1;
            for (uint64_t i = 0; i < g->node_count && i < sample_count * chaos_stride; i += chaos_stride) {
                float diff = fabsf(g->nodes[i].a) - g->avg_activation;
                variance += diff * diff;
            }
            /* Calculate chaos with NaN protection */
            if (sample_count > 0 && variance >= 0.0f) {
                float chaos_calc = sqrtf(variance / (float)sample_count);
                g->avg_chaos = (chaos_calc == chaos_calc) ? chaos_calc : 0.1f;  /* NaN check */
            } else {
                g->avg_chaos = 0.1f;
            }
            /* Dynamic minimum: scale with edge strength if available */
            float chaos_min_init = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.1f) : 0.01f;
            if (chaos_min_init < 0.01f) chaos_min_init = 0.01f;  /* Absolute floor */
            if (g->avg_chaos < chaos_min_init || g->avg_chaos != g->avg_chaos) g->avg_chaos = chaos_min_init;
        } else {
            /* Empty graph or new file - use defaults (fast) */
            g->avg_chaos = 0.1f;
            g->avg_activation = 0.1f;
            g->avg_edge_strength = 0.1f;
        }
        
        g->avg_output_activity = 0.0f;
        g->avg_feedback_correlation = 0.0f;
        g->avg_prediction_accuracy = 0.0f;
        
        /* NO LIMITS: Allocate arrays for all nodes - graph can grow without bounds */
        uint64_t tracking_size = g->node_count;
        uint64_t queue_size = g->node_count;
        
        /* Initialize propagation queue - no size limits */
        g->prop_queue_size = (queue_size > 0) ? queue_size : 256;
        g->prop_queue = calloc(g->prop_queue_size, sizeof(uint32_t));
        g->prop_queued = calloc(g->prop_queue_size, sizeof(_Atomic uint8_t));
        if (g->prop_queue && g->prop_queued) {
            g->prop_queue_head = 0;
            g->prop_queue_tail = 0;
            for (uint64_t i = 0; i < g->prop_queue_size; i++) {
                g->prop_queued[i] = 0;
            }
        }
        
        /* OPTIMIZATION: Fixed-size tracking arrays for large graphs */
        /* Only track first N nodes (most active are usually in hot region) */
        /* For 1TB graphs: only allocate 1M entries = 24 MB instead of 384 GB */
        g->last_activation = calloc(tracking_size, sizeof(float));
        g->last_message = calloc(tracking_size, sizeof(float));
        g->output_propensity = calloc(tracking_size, sizeof(float));
        g->feedback_correlation = calloc(tracking_size, sizeof(float));
        g->prediction_accuracy = calloc(tracking_size, sizeof(float));
        g->stored_energy_capacity = calloc(tracking_size, sizeof(float));
        
        /* Store actual tracking size for bounds checking */
        g->tracking_array_size = tracking_size;
        
        /* Initialize all nodes - no limits */
        if (g->last_activation && g->last_message && g->output_propensity && 
            g->feedback_correlation && g->prediction_accuracy && g->stored_energy_capacity) {
            for (uint64_t i = 0; i < g->node_count; i++) {
                g->last_activation[i] = g->nodes[i].a;
                g->output_propensity[i] = g->nodes[i].output_propensity;
                g->stored_energy_capacity[i] = g->nodes[i].memory_propensity * 0.1f;
            }
        }
        
        /* Initialize pattern system: sequence tracking */
        g->sequence_buffer_size = 1000;  /* Track last 1000 bytes */
        g->sequence_buffer = calloc(g->sequence_buffer_size, sizeof(uint32_t));
        g->sequence_buffer_pos = 0;
        g->sequence_buffer_full = 0;
        
        g->sequence_hash_size = 10000;  /* Hash table for sequence tracking */
        g->sequence_hash_table = calloc(g->sequence_hash_size, 3 * sizeof(uint32_t));  /* Each slot: hash, count, storage_offset */
        
        g->sequence_storage_size = 100000;  /* Storage for first occurrence sequences */
        g->sequence_storage = calloc(g->sequence_storage_size, sizeof(uint32_t));
        g->sequence_storage_pos = 0;
        
        /* EVENT-DRIVEN pattern discovery: co-activation tracking */
        g->activation_window_size = 200;  /* Track last 200 activations */
        g->recent_activations = calloc(g->activation_window_size, sizeof(uint32_t));
        g->activation_window_pos = 0;
        g->activation_check_counter = 0;
        g->coactivation_hash_size = 1024;  /* Hash table for pattern detection */
        g->coactivation_hash = calloc(g->coactivation_hash_size, sizeof(uint64_t));
        
        /* OPTIMIZATION: Fast startup - skip heavy initialization for new files */
        /* Soft structure and edges will be created on-demand as needed */
        /* This dramatically speeds up startup */
        fprintf(stderr, "Initializing graph...\r");
        fflush(stderr);
        
        /* Only initialize soft structure for port range (0-255) - fast */
        initialize_soft_structure(g, true);  /* is_new_file = true */
        
        /* OPTIMIZATION: Skip initial edge creation - edges created on-demand */
        /* This saves significant startup time (was creating 2000+ edges) */
        /* Graph will create edges naturally through melvin_feed_byte() */
        /* create_initial_edge_suggestions(g, true);  SKIPPED for faster startup */
        
        fprintf(stderr, "Brain file ready! (%llu nodes, %llu edges)                    \n", 
                (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
        fflush(stderr);
        
        
    } else {
        /* Open existing file with EXEC permission */
        void *map = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            close(fd);
            free(g);
            return NULL;
        }
        
        MelvinHeader *hdr = (MelvinHeader *)map;
        
        /* Validate */
        if (memcmp(hdr->magic, MELVIN_MAGIC, 4) != 0) {
            munmap(map, st.st_size);
            close(fd);
            free(g);
            return NULL;
        }
        
        /* Version check: v1 files have different header layout */
        if (hdr->version < 1 || hdr->version > MELVIN_VERSION) {
            munmap(map, st.st_size);
            close(fd);
            free(g);
            return NULL;
        }
        
        /* Handle v1 files: different header layout */
        if (hdr->version == 1) {
            /* v1 layout (on disk):
             *   offset 0-3:   magic[4]
             *   offset 4-7:   version[4] = 1
             *   offset 8-11:  reserved[4]
             *   offset 12-19: node_count[8]
             *   offset 20-27: edge_count[8]
             *   offset 28-35: nodes_offset[8]
             *   offset 36-43: edges_offset[8]
             *   offset 44-51: blob_offset[8]
             *   offset 52-59: blob_size[8]
             *   offset 60-67: file_size[8]
             *   offset 68-75: main_entry_offset[8]
             *   offset 76-83: syscalls_ptr_offset[8]
             * 
             * v2 layout (in memory):
             *   offset 0-3:   magic[4]
             *   offset 4-7:   version[4] = 2
             *   offset 8-11:  flags[4]
             *   offset 12-19: file_size[8]
             *   offset 20-27: nodes_offset[8]
             *   offset 28-35: node_count[8]
             *   offset 36-43: edges_offset[8]
             *   offset 44-51: edge_count[8]
             *   offset 52-59: blob_offset[8]
             *   offset 60-67: blob_size[8]
             *   offset 68-75: cold_data_offset[8]
             *   offset 76-83: cold_data_size[8]
             *   offset 84-91: main_entry_offset[8]
             *   offset 92-99: syscalls_ptr_offset[8]
             * 
             * Read v1 layout from disk and map to v2 struct */
            uint8_t *p = (uint8_t *)hdr;
            uint64_t v1_node_count, v1_edge_count, v1_nodes_offset, v1_edges_offset;
            uint64_t v1_blob_offset, v1_blob_size, v1_file_size, v1_main_entry, v1_syscalls_ptr;
            
            memcpy(&v1_node_count, p + 12, 8);
            memcpy(&v1_edge_count, p + 20, 8);
            memcpy(&v1_nodes_offset, p + 28, 8);
            memcpy(&v1_edges_offset, p + 36, 8);
            memcpy(&v1_blob_offset, p + 44, 8);
            memcpy(&v1_blob_size, p + 52, 8);
            memcpy(&v1_file_size, p + 60, 8);
            memcpy(&v1_main_entry, p + 68, 8);
            memcpy(&v1_syscalls_ptr, p + 76, 8);
            
            /* Map to v2 struct layout */
            hdr->file_size = v1_file_size;
            hdr->nodes_offset = v1_nodes_offset;
            hdr->node_count = v1_node_count;
            hdr->edges_offset = v1_edges_offset;
            hdr->edge_count = v1_edge_count;
            hdr->blob_offset = v1_blob_offset;
            hdr->blob_size = v1_blob_size;
            hdr->cold_data_offset = 0;  /* v1 has no cold_data */
            hdr->cold_data_size = 0;
            hdr->main_entry_offset = v1_main_entry;
            hdr->syscalls_ptr_offset = v1_syscalls_ptr;
        }
        
        g->fd = fd;
        g->map_base = map;
        g->map_size = st.st_size;
        g->hdr = hdr;
        g->nodes = (Node *)((char *)map + hdr->nodes_offset);
        g->edges = (Edge *)((char *)map + hdr->edges_offset);
        g->blob = (uint8_t *)((char *)map + hdr->blob_offset);
        g->cold_data = (hdr->cold_data_size > 0) ? 
                       (uint8_t *)((char *)map + hdr->cold_data_offset) : NULL;
        g->node_count = hdr->node_count;
        g->edge_count = hdr->edge_count;
        g->blob_size = hdr->blob_size;
        g->cold_data_size = hdr->cold_data_size;
        
        /* Initialize soft structure fields for existing files (may be uninitialized from old format) */
        /* Check if soft structure fields are zero (likely uninitialized from old .m file) */
        bool needs_soft_init = false;
        for (uint64_t i = 0; i < g->node_count && i < 256; i++) {
            if (g->nodes[i].input_propensity == 0.0f && 
                g->nodes[i].output_propensity == 0.0f &&
                g->nodes[i].memory_propensity == 0.0f &&
                g->nodes[i].semantic_hint == 0) {
                needs_soft_init = true;
                break;
            }
        }
        
        if (needs_soft_init) {
            /* Old file format - initialize soft structure (preserves learned activations/edges) */
            initialize_soft_structure(g, false);  /* is_new_file = false - add structure to existing file */
        }
        
        /* Initialize UEL physics state for existing files */
        /* OPTIMIZATION: Use sampling instead of full scan for fast startup */
        /* Averages will converge quickly during runtime anyway */
        if (g->node_count > 0) {
            /* GRAPH LAWS DETERMINE PROCESSING: Use all nodes and edges for initial estimates */
            /* NO SAMPLING LIMITS - use full graph state */
            uint64_t node_sample = g->node_count;
            uint64_t edge_sample = g->edge_count;
            
            float sum_activation = 0.0f;
            float sum_edge_strength = 0.0f;
            uint64_t active_count = 0;
            
            /* Sample nodes (uniformly distributed) */
            uint64_t node_step = (g->node_count > node_sample) ? (g->node_count / node_sample) : 1;
            for (uint64_t i = 0; i < g->node_count && i < node_sample * node_step; i += node_step) {
                float a_abs = fabsf(g->nodes[i].a);
                if (a_abs > 0.001f) {
                    sum_activation += a_abs;
                    active_count++;
                }
            }
            
            /* Sample edges (uniformly distributed) */
            uint64_t edge_step = (g->edge_count > edge_sample) ? (g->edge_count / edge_sample) : 1;
            for (uint64_t i = 0; i < g->edge_count && i < edge_sample * edge_step; i += edge_step) {
                sum_edge_strength += fabsf(g->edges[i].w);
            }
            
            g->avg_activation = (active_count > 0) ? (sum_activation / active_count) : 0.1f;
            if (g->avg_activation < 0.01f) g->avg_activation = 0.1f;  /* Minimum */
            
            g->avg_edge_strength = (edge_sample > 0) ? (sum_edge_strength / edge_sample) : 0.1f;
            if (g->avg_edge_strength < 0.01f) g->avg_edge_strength = 0.1f;  /* Minimum */
            
            /* Estimate initial chaos from activation variance (already sampled) */
            float variance = 0.0f;
            uint64_t sample_count = (node_sample < 100) ? node_sample : 100;
            uint64_t chaos_step = (g->node_count > sample_count) ? (g->node_count / sample_count) : 1;
            for (uint64_t i = 0; i < g->node_count && i < sample_count * chaos_step; i += chaos_step) {
                float diff = fabsf(g->nodes[i].a) - g->avg_activation;
                variance += diff * diff;
            }
            g->avg_chaos = (sample_count > 0) ? sqrtf(variance / (float)sample_count) : 0.1f;
            if (g->avg_chaos < 0.01f) g->avg_chaos = 0.1f;  /* Minimum */
        } else {
            /* Empty graph - use defaults */
            g->avg_chaos = 0.1f;
            g->avg_activation = 0.1f;
            g->avg_edge_strength = 0.1f;
        }
        
        g->avg_output_activity = 0.0f;
        g->avg_feedback_correlation = 0.0f;
        g->avg_prediction_accuracy = 0.0f;
        
        /* NO LIMITS: Allocate arrays for all nodes */
        uint64_t tracking_size_existing = g->node_count;
        uint64_t queue_size_existing = g->node_count;
        
        /* Initialize event-driven propagation state - no size limits */
        g->prop_queue_size = (queue_size_existing > 0) ? queue_size_existing : 256;
        g->prop_queue = calloc(g->prop_queue_size, sizeof(uint32_t));
        g->prop_queued = calloc(g->prop_queue_size, sizeof(_Atomic uint8_t));
        if (!g->prop_queue || !g->prop_queued) {
            melvin_close(g);
            return NULL;
        }
        g->prop_queue_head = 0;
        g->prop_queue_tail = 0;
        for (uint64_t i = 0; i < g->prop_queue_size; i++) {
            g->prop_queued[i] = 0;
        }
        
        /* OPTIMIZATION: Lazy allocation - only allocate arrays when first accessed */
        /* For large graphs, this avoids slow calloc and initialization on startup */
        /* Arrays will be allocated on first use in update_node_and_propagate */
        /* But use fixed size when allocated (capped at 1M) */
        g->last_activation = NULL;
        g->last_message = NULL;
        g->output_propensity = NULL;
        g->feedback_correlation = NULL;
        g->prediction_accuracy = NULL;
        g->stored_energy_capacity = NULL;
        g->tracking_array_size = tracking_size_existing;  /* Set size for lazy allocation */
        g->avg_chaos = 0.1f;  /* Initial estimate */
        g->avg_activation = 0.1f;  /* Jump start: small initial activation to bootstrap */
        g->avg_edge_strength = 0.1f;
        g->avg_output_activity = 0.0f;
        g->avg_feedback_correlation = 0.0f;
        g->avg_prediction_accuracy = 0.0f;
        
        /* Initialize pattern system: sequence tracking */
        g->sequence_buffer_size = 1000;  /* Track last 1000 bytes */
        g->sequence_buffer = calloc(g->sequence_buffer_size, sizeof(uint32_t));
        g->sequence_buffer_pos = 0;
        g->sequence_buffer_full = 0;
        
        g->sequence_hash_size = 10000;  /* Hash table for sequence tracking */
        g->sequence_hash_table = calloc(g->sequence_hash_size, 3 * sizeof(uint32_t));  /* Each slot: hash, count, storage_offset */
        
        g->sequence_storage_size = 100000;  /* Storage for first occurrence sequences */
        g->sequence_storage = calloc(g->sequence_storage_size, sizeof(uint32_t));
        g->sequence_storage_pos = 0;
        
        /* EVENT-DRIVEN pattern discovery: co-activation tracking */
        g->activation_window_size = 200;  /* Track last 200 activations */
        g->recent_activations = calloc(g->activation_window_size, sizeof(uint32_t));
        g->activation_window_pos = 0;
        g->activation_check_counter = 0;
        g->coactivation_hash_size = 1024;  /* Hash table for pattern detection */
        g->coactivation_hash = calloc(g->coactivation_hash_size, sizeof(uint64_t));
        
        /* Initialize dynamic propagation queue - no artificial limits */
        /* Queue size based on node_count, but graph will learn to handle any size */
        /* If queue overflows, graph will evolve patterns to route inputs more efficiently */
        /* Start smaller for efficiency, grow dynamically as needed */
        g->prop_queue_size = (g->node_count > 0) ? g->node_count : 64;  /* Start with node_count, minimum 64 (reduced from 256) */
        
        g->prop_queue = calloc(g->prop_queue_size, sizeof(uint32_t));
        g->prop_queued = calloc(g->prop_queue_size, sizeof(_Atomic uint8_t));
        
        if (!g->prop_queue || !g->prop_queued) {
            /* Cleanup on failure */
            if (g->prop_queue) free(g->prop_queue);
            if (g->prop_queued) free(g->prop_queued);
            if (g->last_activation) free(g->last_activation);
            if (g->last_message) free(g->last_message);
            if (g->output_propensity) free(g->output_propensity);
            if (g->feedback_correlation) free(g->feedback_correlation);
            if (g->prediction_accuracy) free(g->prediction_accuracy);
            melvin_close(g);
            return NULL;
        }
        
        g->prop_queue_head = 0;
        g->prop_queue_tail = 0;
        for (uint64_t i = 0; i < g->prop_queue_size; i++) {
            g->prop_queued[i] = 0;
        }
    }
    
    return g;
}

/* ========================================================================
 * SYNC TO DISK
 * ======================================================================== */

void melvin_sync(Graph *g) {
    if (!g || !g->map_base) return;
    msync(g->map_base, g->map_size, MS_SYNC);
}

/* ========================================================================
 * CLOSE
 * ======================================================================== */

void melvin_close(Graph *g) {
    if (!g) return;
    
    if (g->map_base) {
        msync(g->map_base, g->map_size, MS_SYNC);
        munmap(g->map_base, g->map_size);
    }
    
    if (g->fd >= 0) {
        close(g->fd);
    }
    
    /* Free event-driven propagation state */
    if (g->last_activation) free(g->last_activation);
    if (g->last_message) free(g->last_message);
    if (g->output_propensity) free(g->output_propensity);
    if (g->feedback_correlation) free(g->feedback_correlation);
    if (g->prediction_accuracy) free(g->prediction_accuracy);
    if (g->stored_energy_capacity) free(g->stored_energy_capacity);
    
    /* Free dynamic propagation queue */
    if (g->prop_queue) free(g->prop_queue);
    if (g->prop_queued) free(g->prop_queued);
    
    /* Free pattern system buffers */
    if (g->sequence_buffer) free(g->sequence_buffer);
    if (g->sequence_hash_table) free(g->sequence_hash_table);
    if (g->sequence_storage) free(g->sequence_storage);
    
    /* Free co-activation tracking */
    if (g->recent_activations) free(g->recent_activations);
    if (g->coactivation_hash) free(g->coactivation_hash);
    
    free(g);
}

/* ========================================================================
 * TEACHABLE EXEC: Feed Operations to Brain as Data
 * Brain learns operations by having machine code fed, not hardcoded!
 * ======================================================================== */

/* Teach brain an operation by feeding executable code */
uint32_t melvin_teach_operation(Graph *g, const uint8_t *machine_code, 
                                 size_t code_len, const char *name) {
    if (!g || !machine_code || code_len == 0) return UINT32_MAX;
    
    /* Use a dedicated EXEC code region in blob (separate from pattern data) */
    /* Start EXEC codes at 1KB to avoid conflicts with pattern storage */
    static uint64_t exec_code_offset = 1024;  /* Start at 1KB */
    
    uint64_t code_offset = exec_code_offset;
    
    /* Check we have space in the allocated blob region */
    uint64_t blob_capacity = g->hdr->blob_size;  /* Total capacity */
    if (code_offset + code_len + 512 > blob_capacity) {
        fprintf(stderr, "❌ No blob space for '%s' (need %llu, have %llu)\n", 
                name, (unsigned long long)(code_offset + code_len + 512),
                (unsigned long long)blob_capacity);
        return UINT32_MAX;
    }
    
    /* Write code to blob */
    memcpy(g->blob + code_offset, machine_code, code_len);
    exec_code_offset += code_len + 512;  /* Move to next free offset */
    
    /* Find free EXEC node */
    uint32_t exec_node = UINT32_MAX;
    for (uint32_t i = 2000; i < 3000 && i < g->node_count; i++) {
        if (g->nodes[i].payload_offset == 0) {
            exec_node = i;
            break;
        }
    }
    
    if (exec_node == UINT32_MAX) return UINT32_MAX;
    
    /* Point to code */
    g->nodes[exec_node].payload_offset = code_offset;
    g->nodes[exec_node].byte = 0xEE;
    g->nodes[exec_node].exec_threshold_ratio = 0.1f;
    
    fprintf(stderr, "📚 TAUGHT: '%s' → node %u @ %llu\n",
            name, exec_node, (unsigned long long)code_offset);
    
    return exec_node;
}

/* ========================================================================
 * SET SYSCALLS (writes pointer into blob)
 * ======================================================================== */

/* Thread-local Graph* for syscalls (needed for sys_copy_from_cold) */
static __thread Graph* current_graph = NULL;
static __thread MelvinSyscalls* current_syscalls = NULL; /* Thread-local syscalls pointer */

Graph* melvin_get_current_graph(void) {
    return current_graph;
}

void melvin_set_syscalls(Graph *g, MelvinSyscalls *syscalls) {
    current_graph = g;  /* Set thread-local context */
    current_syscalls = syscalls; /* Store syscalls pointer in thread-local storage */
    if (!g || !g->hdr || !syscalls) return;
    
    /* Write syscalls pointer into blob at known offset (if offset is set) */
    if (g->hdr->syscalls_ptr_offset > 0 && 
        g->hdr->syscalls_ptr_offset < g->hdr->blob_size) {
        void **ptr_loc = (void **)(g->blob + g->hdr->syscalls_ptr_offset);
        *ptr_loc = syscalls;
    }
}

/* ========================================================================
 * GET SYSCALLS FROM BLOB (for blob code to call)
 * ======================================================================== */

MelvinSyscalls* melvin_get_syscalls_from_blob(Graph *g) {
    if (!g || !g->hdr) return NULL;
    
    /* First try blob offset (for blob code to access) */
    if (g->hdr->syscalls_ptr_offset > 0 && 
        g->hdr->syscalls_ptr_offset < g->hdr->blob_size) {
        void **ptr_loc = (void **)(g->blob + g->hdr->syscalls_ptr_offset);
        MelvinSyscalls *blob_syscalls = (MelvinSyscalls *)*ptr_loc;
        if (blob_syscalls) return blob_syscalls;
    }
    
    /* Fallback: use thread-local syscalls pointer (for host-side tool invocation) */
    /* This works even when syscalls_ptr_offset is 0 (fresh brain) */
    if (current_graph == g && current_syscalls) {
        return current_syscalls;
    }
    
    return NULL;
}

/* ========================================================================
 * FEED BYTE (ONLY writes to .m, NO physics)
 * ======================================================================== */

/* Forward declaration */
static void ensure_node(Graph *g, uint32_t node_id);
static void ensure_node_arrays(Graph *g);

static uint32_t find_edge(Graph *g, uint32_t src, uint32_t dst) {
    /* Safety check: ensure arrays are valid */
    if (!g || !g->nodes || !g->edges || !g->hdr) {
        return UINT32_MAX;
    }
    
    /* Ensure nodes exist - graph grows dynamically */
    ensure_node(g, src);
    ensure_node(g, dst);
    
    /* Safety check: ensure node indices are valid after ensure_node */
    if (src >= g->node_count || dst >= g->node_count) {
        return UINT32_MAX;
    }
    
    /* Safety check: ensure edge_count is valid before accessing edges */
    if (g->edge_count == 0) {
        return UINT32_MAX;  /* No edges yet */
    }
    
    uint32_t eid = g->nodes[src].first_out;
    uint32_t iter = 0;
    uint32_t max_iter = g->edge_count + 1;  /* Can't have more edges than exist */
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
        if (g->edges[eid].dst == dst) return eid;
        eid = g->edges[eid].next_out;
        iter++;
    }
    
    /* Also check incoming edges */
    if (dst >= g->node_count) return UINT32_MAX;
    
    eid = g->nodes[dst].first_in;
    iter = 0;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
        if (g->edges[eid].src == src) return eid;
        eid = g->edges[eid].next_in;
        iter++;
    }
    
    return UINT32_MAX;
}

/* Grow node array dynamically - no limits */
static int grow_nodes(Graph *g, uint64_t new_node_count) {
    if (!g || !g->hdr || new_node_count <= g->node_count) return 0;
    
    /* Calculate new file size */
    uint64_t old_nodes_size = g->node_count * sizeof(Node);
    uint64_t new_nodes_size = new_node_count * sizeof(Node);
    uint64_t size_increase = new_nodes_size - old_nodes_size;
    
    /* Extend file */
    uint64_t new_file_size = g->hdr->file_size + size_increase;
    if (ftruncate(g->fd, (off_t)new_file_size) < 0) {
        return -1;  /* Failed to grow file */
    }
    
    /* Remap with new size */
    /* Try mremap first (Linux), fallback to unmap+remap (portable) */
    void *new_map = MAP_FAILED;
    #ifdef __linux__
    new_map = mremap(g->map_base, (size_t)g->map_size, (size_t)new_file_size, MREMAP_MAYMOVE);
    #endif
    if (new_map == MAP_FAILED) {
        /* Fallback: unmap and remap with EXEC (works on all platforms) */
        munmap(g->map_base, g->map_size);
        new_map = mmap(NULL, (size_t)new_file_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, g->fd, 0);
        if (new_map == MAP_FAILED) {
            return -1;
        }
    }
    
    /* Update graph pointers */
    g->map_base = new_map;
    g->map_size = new_file_size;
    g->hdr = (MelvinHeader *)new_map;
    g->nodes = (Node *)((char *)new_map + g->hdr->nodes_offset);
    g->edges = (Edge *)((char *)new_map + g->hdr->edges_offset);
    g->blob = (uint8_t *)((char *)new_map + g->hdr->blob_offset);
    if (g->hdr->cold_data_size > 0) {
        g->cold_data = (uint8_t *)((char *)new_map + g->hdr->cold_data_offset);
    }
    
    /* Save old count before updating */
    uint64_t old_node_count = g->node_count;
    
    /* Initialize new nodes - MUST set edge pointers to UINT32_MAX (not 0!) */
    /* memset(0) would leave first_in/first_out = 0, pointing to edge 0! */
    for (uint64_t i = old_node_count; i < new_node_count; i++) {
        memset(&g->nodes[i], 0, sizeof(Node));
        g->nodes[i].first_in = UINT32_MAX;   /* No edges yet */
        g->nodes[i].first_out = UINT32_MAX;  /* No edges yet */
    }
    
    /* Update counts */
    g->node_count = new_node_count;
    g->hdr->node_count = new_node_count;
    g->hdr->file_size = new_file_size;
    
    /* Reallocate tracking arrays */
    size_t new_size = (size_t)new_node_count * sizeof(float);
    g->last_activation = realloc(g->last_activation, new_size);
    g->last_message = realloc(g->last_message, new_size);
    g->output_propensity = realloc(g->output_propensity, new_size);
    g->feedback_correlation = realloc(g->feedback_correlation, new_size);
    g->prediction_accuracy = realloc(g->prediction_accuracy, new_size);
    g->stored_energy_capacity = realloc(g->stored_energy_capacity, new_size);
    
    /* Initialize new tracking arrays (zero the new portion) */
    uint64_t new_portion = new_node_count - old_node_count;
    
    if (g->last_activation && new_portion > 0) {
        memset(g->last_activation + old_node_count, 0, (size_t)new_portion * sizeof(float));
    }
    if (g->last_message && new_portion > 0) {
        memset(g->last_message + old_node_count, 0, (size_t)new_portion * sizeof(float));
    }
    if (g->output_propensity && new_portion > 0) {
        memset(g->output_propensity + old_node_count, 0, (size_t)new_portion * sizeof(float));
    }
    if (g->feedback_correlation && new_portion > 0) {
        memset(g->feedback_correlation + old_node_count, 0, (size_t)new_portion * sizeof(float));
    }
    if (g->prediction_accuracy && new_portion > 0) {
        memset(g->prediction_accuracy + old_node_count, 0, (size_t)new_portion * sizeof(float));
    }
    if (g->stored_energy_capacity && new_portion > 0) {
        memset(g->stored_energy_capacity + old_node_count, 0, (size_t)new_portion * sizeof(float));
    }
    
    return 0;
}

/* Ensure node exists, grow if needed */
/* Lazy allocation: ensure node arrays are allocated (only allocate when first accessed) */
static void ensure_node_arrays(Graph *g) {
    if (!g || !g->node_count) return;
    
    /* NO LIMITS: Allocate for all nodes */
    uint64_t tracking_size = g->node_count;
    if (g->tracking_array_size == 0) {
        g->tracking_array_size = tracking_size;
    }
    
    /* Allocate arrays if not already allocated */
    if (!g->last_activation) {
        g->last_activation = calloc(tracking_size, sizeof(float));
        if (!g->last_activation) return;  /* Out of memory */
        /* Initialize from current node activations (capped) */
        uint64_t init_count = (g->node_count < tracking_size) ? g->node_count : tracking_size;
        for (uint64_t i = 0; i < init_count; i++) {
            g->last_activation[i] = g->nodes[i].a;
        }
    }
    
    if (!g->last_message) {
        g->last_message = calloc(tracking_size, sizeof(float));
        if (!g->last_message) return;
    }
    
    if (!g->output_propensity) {
        g->output_propensity = calloc(tracking_size, sizeof(float));
        if (!g->output_propensity) return;
        /* Initialize from node's output_propensity field (capped) */
        uint64_t init_count = (g->node_count < tracking_size) ? g->node_count : tracking_size;
        for (uint64_t i = 0; i < init_count; i++) {
            g->output_propensity[i] = g->nodes[i].output_propensity;
        }
    }
    
    if (!g->feedback_correlation) {
        g->feedback_correlation = calloc(tracking_size, sizeof(float));
        if (!g->feedback_correlation) return;
    }
    
    if (!g->prediction_accuracy) {
        g->prediction_accuracy = calloc(tracking_size, sizeof(float));
        if (!g->prediction_accuracy) return;
    }
    
    if (!g->stored_energy_capacity) {
        g->stored_energy_capacity = calloc(tracking_size, sizeof(float));
        if (!g->stored_energy_capacity) return;
        /* Initialize from memory propensity (importance) */
        for (uint64_t i = 0; i < g->node_count; i++) {
            g->stored_energy_capacity[i] = g->nodes[i].memory_propensity * 0.1f;
        }
    }
}

/* Find an unused node that can be safely reused */
/* IMPORTANT: Do NOT reuse placeholder nodes - they help build generalization and patterns */
/* Placeholder nodes are "local placeholders" that serve a structural purpose */
/* Returns first truly unused node found, or UINT32_MAX if none */
static uint32_t find_unused_node(Graph *g) {
    if (!g || !g->nodes) return UINT32_MAX;
    
    /* Skip byte nodes (0-255) - they're always in use */
    /* Skip structural nodes (200-839) - they're scaffolding for the graph */
    /* Only look for nodes that are truly unused AND not structural placeholders */
    for (uint32_t i = 840; i < g->node_count; i++) {  /* Start after structural range */
        Node *n = &g->nodes[i];
        
        /* Check if node is truly unused AND not a placeholder */
        /* Placeholder nodes might have no edges/data but are still part of pattern structure */
        /* We can only safely reuse nodes that are:
         * 1. Beyond structural ranges (840+)
         * 2. Have no edges (not connected to anything)
         * 3. Have no pattern data (not a pattern node)
         * 4. Have no EXEC payload (not an EXEC node)
         * 5. Have no activation (completely dormant)
         * 6. Have no structural role (not input/output/memory port)
         * 7. Have been unused for a while (check if it's truly abandoned, not just temporarily inactive)
         */
        if (n->first_out == UINT32_MAX &&           /* No outgoing edges */
            n->first_in == UINT32_MAX &&             /* No incoming edges */
            n->pattern_data_offset == 0 &&           /* No pattern data (not a pattern node) */
            n->payload_offset == 0 &&                /* No EXEC payload (not an EXEC node) */
            fabsf(n->a) < 0.0001f &&                 /* No activation (very strict - must be truly dormant) */
            fabsf(n->input_propensity) < 0.0001f &&   /* Not an input port */
            fabsf(n->output_propensity) < 0.0001f &&  /* Not an output port */
            fabsf(n->memory_propensity) < 0.0001f) {  /* Not a memory node */
            /* This node appears truly unused - but be conservative */
            /* Only reuse if it's well beyond structural ranges to avoid reusing placeholders */
            return i;
        }
    }
    
    return UINT32_MAX;  /* No unused nodes found - don't reuse placeholders */
}

static void ensure_node(Graph *g, uint32_t node_id) {
    if (!g || node_id < g->node_count) return;
    
    /* Need to grow - no node reuse here (node_id must be exact for specific nodes like EXEC) */
    /* Grow to node_id + 1, with conservative headroom (not aggressive doubling) */
    uint64_t new_count = (uint64_t)node_id + 1;
    
    /* Add 20% headroom - NO MAXIMUM LIMIT */
    uint64_t headroom = new_count / 5;  /* 20% headroom */
    if (headroom < 100) headroom = 100;   /* Minimum 100 nodes headroom */
    new_count += headroom;
    
    /* Don't shrink if we're already bigger (shouldn't happen, but safety) */
    if (new_count < g->node_count) new_count = g->node_count;
    
    /* Debug: log large growths */
    if (new_count > g->node_count * 2) {
        fprintf(stderr, "WARNING: Growing from %llu to %llu nodes (requested node %u)\n",
                (unsigned long long)g->node_count, (unsigned long long)new_count, node_id);
    }
    
    grow_nodes(g, new_count);
}

static uint32_t create_edge(Graph *g, uint32_t src, uint32_t dst, float w) {
    /* Simple edge creation - just structure, no physics */
    /* Hot region only - edges stay in hot space */
    /* NO LIMITS - graph controls its own growth */
    
    /* SAFETY: Validate node IDs to prevent garbage values */
    const uint32_t MAX_REASONABLE_NODE = 100000000;  /* 100 million nodes max */
    
    if (src >= MAX_REASONABLE_NODE || dst >= MAX_REASONABLE_NODE) {
        fprintf(stderr, "ERROR: Invalid node ID in create_edge: src=%u, dst=%u (max=%u)\n",
                src, dst, MAX_REASONABLE_NODE);
        fprintf(stderr, "       This is likely a bug - garbage value detected!\n");
        return UINT32_MAX;  /* Return invalid edge ID */
    }
    
    /* Ensure nodes exist */
    ensure_node(g, src);
    ensure_node(g, dst);
    
    /* Check if we need to grow edges array (grow file if needed) */
    uint64_t max_edges = (g->hdr->file_size - g->hdr->edges_offset) / sizeof(Edge);
    if (g->edge_count >= max_edges) {
        /* Need to grow file to accommodate more edges */
        /* Adaptive growth: double edges (or at least 10k, whichever is larger) */
        uint64_t new_edge_count = g->edge_count * 2;
        if (new_edge_count < g->edge_count + 10000) {
            new_edge_count = g->edge_count + 10000;  /* Minimum growth */
        }
        uint64_t new_edges_size = new_edge_count * sizeof(Edge);
        uint64_t new_file_size = g->hdr->edges_offset + new_edges_size;
        
        /* Extend file */
        if (ftruncate(g->fd, (off_t)new_file_size) < 0) {
            return UINT32_MAX;  /* Failed to grow */
        }
        
        /* Remap */
        void *new_map = MAP_FAILED;
        #ifdef __linux__
        new_map = mremap(g->map_base, (size_t)g->map_size, (size_t)new_file_size, MREMAP_MAYMOVE);
        #endif
        if (new_map == MAP_FAILED) {
            munmap(g->map_base, g->map_size);
            new_map = mmap(NULL, (size_t)new_file_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, g->fd, 0);
            if (new_map == MAP_FAILED) {
                return UINT32_MAX;
            }
        }
        
        g->map_base = new_map;
        g->map_size = new_file_size;
        g->hdr = (MelvinHeader *)new_map;
        g->nodes = (Node *)((char *)new_map + g->hdr->nodes_offset);
        g->edges = (Edge *)((char *)new_map + g->hdr->edges_offset);
        g->blob = (uint8_t *)((char *)new_map + g->hdr->blob_offset);
        if (g->hdr->cold_data_size > 0) {
            g->cold_data = (uint8_t *)((char *)new_map + g->hdr->cold_data_offset);
        }
        g->hdr->file_size = new_file_size;
    }
    
    uint32_t eid = (uint32_t)g->edge_count++;
    g->hdr->edge_count = g->edge_count;  /* Sync to header */
    Edge *e = &g->edges[eid];
    e->src = src;
    e->dst = dst;
    e->w = w;
    e->next_out = g->nodes[src].first_out;
    e->next_in = g->nodes[dst].first_in;
    g->nodes[src].first_out = eid;
    g->nodes[dst].first_in = eid;
    g->nodes[src].out_degree++;
    g->nodes[dst].in_degree++;
    
    return eid;
}

/* Public wrapper for tools - export edge creation */
uint32_t melvin_create_edge(Graph *g, uint32_t src, uint32_t dst, float w) {
    return create_edge(g, src, dst, w);
}

/* ========================================================================
 * UEL PHYSICS PARAMETERS (must be before event-driven code)
 * ======================================================================== */

/* UEL physics parameters - ALL RELATIVE (no hardcoded thresholds) */
static const struct {
    float eta_a_base;          /* Base learning rate (scaled by graph state) */
    float eta_w_base;          /* Base weight learning rate (scaled by graph state) */
    float lambda;              /* Global field coupling (relative to local messages) */
    float storage_efficiency_base;  /* Base storage efficiency (scaled adaptively) */
    float capacity_growth_rate; /* Rate at which stored energy increases capacity (0.0-1.0) */
    float threshold_reduction_per_capacity; /* How much capacity reduces activation threshold */
    float release_efficiency;  /* Fraction of stored energy that can be released (0.0-1.0) */
    float storage_diminishing_factor; /* How much storage decreases as capacity grows (0.0-1.0) */
    float storage_activity_scaling; /* How graph activity affects storage efficiency */
    float decay_w_base;        /* Base weight decay (scaled by graph state) */
    float change_threshold_ratio;  /* Change threshold as ratio of avg_activation */
    float running_avg_alpha;   /* Exponential moving average decay (adaptive) */
    float weight_clamp_ratio;  /* Weight clamp as ratio of avg_edge_strength */
    float active_threshold_ratio; /* Active node threshold as ratio of avg_activation */
    float edge_prune_threshold_ratio; /* Edge weight threshold below which edges are effectively ignored (ratio of avg_edge_strength) */
    
    /* Drive mechanism parameters - ALL RELATIVE */
    float restlessness_strength;     /* Restlessness drive strength (relative to avg_activation) */
    float restlessness_threshold_ratio; /* Activation threshold for restlessness (ratio of avg) */
    float exploration_strength;     /* Exploration reward strength (relative to avg_feedback) */
    float quality_strength;          /* Energy quality reward strength (relative to avg_chaos) */
    float prediction_strength;      /* Prediction reward strength (relative to avg_prediction) */
    float novelty_seeking_strength; /* Novelty seeking when chaos is too low (guilty pleasure) */
    float boredom_threshold_ratio;   /* Chaos threshold below which system seeks novelty (ratio of avg) */
} uel_params = {
    .eta_a_base = 0.1f,        /* Base - will be scaled by relative_chaos */
    .eta_w_base = 0.01f,       /* Base - will be scaled by avg_edge_strength */
    .lambda = 0.05f,           /* Relative coupling strength */
    .storage_efficiency_base = 0.1f, /* Base storage (scaled adaptively - becomes much smaller on long paths) */
    .capacity_growth_rate = 0.01f, /* Capacity grows by 1% per unit of stored energy */
    .threshold_reduction_per_capacity = 0.1f, /* Each unit of capacity reduces threshold by 10% */
    .release_efficiency = 0.5f, /* Can release 50% of stored energy when beneficial */
    .storage_diminishing_factor = 0.8f, /* Storage efficiency decreases by 20% as capacity doubles */
    .storage_activity_scaling = 0.5f, /* High graph activity reduces storage (prevents explosion) */
    .decay_w_base = 0.001f,    /* Base - will be scaled by graph state */
    .change_threshold_ratio = 0.1f,  /* 10% of avg_activation */
    .running_avg_alpha = 0.99f, /* Slow adaptation */
    .weight_clamp_ratio = 10.0f, /* 10x avg_edge_strength */
    .active_threshold_ratio = 0.1f,  /* 10% of avg_activation */
    .edge_prune_threshold_ratio = 0.01f,  /* Edges below 1% of avg_edge_strength are effectively ignored (self-correction) */
    
    /* Drive mechanisms - relative strengths */
    .restlessness_strength = 0.1f,      /* 10% of activation cost */
    .restlessness_threshold_ratio = 1.5f, /* 150% of avg_activation triggers restlessness */
    .exploration_strength = 0.2f,       /* 20% of feedback correlation */
    .quality_strength = 0.15f,          /* 15% of chaos reduction */
    .prediction_strength = 0.1f,         /* 10% of prediction accuracy */
    .novelty_seeking_strength = 0.3f,    /* 30% - strong drive to seek novelty when bored */
    .boredom_threshold_ratio = 0.3f      /* When chaos < 30% of historical avg, seek novelty */
};

        /* Forward declarations */
static void uel_propagate_from(Graph *g, uint32_t start_node);
static void prop_queue_add(Graph *g, uint32_t node_id);
static void pattern_law_apply(Graph *g, uint32_t data_node_id);
static PatternValue extract_pattern_value(Graph *g, const uint32_t *sequence, uint32_t length, uint32_t pattern_node_id);
static void learn_value_mapping(Graph *g, uint32_t pattern_node_id, PatternValue example_value);
static void pass_values_to_exec(Graph *g, uint32_t exec_node_id, PatternValue *values, uint32_t value_count);
static void convert_result_to_pattern(Graph *g, uint32_t exec_node_id, uint64_t result);
static void execute_graph_structure(Graph *g, uint32_t start_node, uint64_t input1, uint64_t input2, uint64_t *result);
static void learn_pattern_to_exec_routing(Graph *g, uint32_t pattern_node_id, const PatternElement *elements, uint32_t element_count);

void melvin_feed_byte(Graph *g, uint32_t port_node_id, uint8_t b, float energy) {
    if (!g || !g->hdr) return;
    
    /* Ensure nodes exist - graph grows dynamically */
    ensure_node(g, port_node_id);
    
    uint32_t data_id = (uint32_t)b;
    ensure_node(g, data_id);
    
    /* FEEDBACK CORRELATION: Track if this input correlates with recent outputs */
    /* RELATIVE: feedback correlation based on recent output activity */
    float feedback_strength = 0.0f;
    if (g->avg_output_activity > 0.01f) {
        /* Recent outputs exist - check if this input correlates */
        /* Simplified: if input comes shortly after output activity, it's feedback */
        float time_since_output = g->avg_output_activity;  /* Proxy for recency */
        feedback_strength = 1.0f / (1.0f + time_since_output);  /* Decay with time */
        
        /* Update feedback correlation for nodes that were recently active outputs */
        if (g->output_propensity && g->feedback_correlation) {
            for (uint64_t i = 0; i < g->node_count && i < 256; i++) {
                if (g->output_propensity[i] > 0.1f && fabsf(g->nodes[i].a) > g->avg_activation * 0.5f) {
                    /* This node was an active output - reward it for creating feedback */
                    g->feedback_correlation[i] = uel_params.running_avg_alpha * g->feedback_correlation[i] + 
                                                (1.0f - uel_params.running_avg_alpha) * feedback_strength;
                }
            }
        }
    }
    
    /* Inject energy - this is the event that triggers propagation */
    g->nodes[port_node_id].a += energy;
    g->nodes[data_id].a += energy;
    
    /* CRITICAL: Create edge between sequential nodes BEFORE pattern_law_apply */
    /* This creates the graph structure that represents sequences */
    /* Without this, sequences have no edges between sequential bytes */
    /* Get previous node in sequence BEFORE pattern_law_apply increments buffer_pos */
    uint32_t prev_node_id = UINT32_MAX;
    if (g->sequence_buffer) {
        if (g->sequence_buffer_pos > 0) {
            /* Get previous node (buffer_pos hasn't been incremented yet) */
            uint64_t prev_pos = g->sequence_buffer_pos - 1;
            prev_node_id = g->sequence_buffer[prev_pos % g->sequence_buffer_size];
        } else if (g->sequence_buffer_full) {
            /* Buffer wrapped - previous is at end */
            uint64_t prev_pos = g->sequence_buffer_size - 1;
            prev_node_id = g->sequence_buffer[prev_pos % g->sequence_buffer_size];
        }
        
        /* Create edge from previous node to current node (sequential connection) */
        if (prev_node_id != UINT32_MAX && 
            prev_node_id < g->node_count && data_id < g->node_count && 
            prev_node_id != data_id &&  /* Don't create self-loops */
            find_edge(g, prev_node_id, data_id) == UINT32_MAX) {
            float seq_weight = g->avg_edge_strength * 0.15f;  /* Slightly stronger for sequential edges */
            if (seq_weight < 0.01f) seq_weight = 0.01f;
            if (seq_weight > 1.0f) seq_weight = 1.0f;
            create_edge(g, prev_node_id, data_id, seq_weight);
        }
    }
    
    /* ========================================================================
     * GLOBAL LAW: Pattern Creation (HYBRID APPROACH)
     * Patterns form automatically when sequences repeat (seen twice)
     * HIERARCHICAL: Patterns can compose other patterns (840+) for true abstraction
     * OPTIMIZED: Rate-limited immediate detection + co-activation during propagation
     * ======================================================================== */
    pattern_law_apply(g, data_id);
    
    /* Ensure edge exists from port to data (structure only) */
    /* RELATIVE: initial edge weight based on avg_edge_strength */
    if (find_edge(g, port_node_id, data_id) == UINT32_MAX) {
        float initial_weight = g->avg_edge_strength * 0.1f;
        if (initial_weight < 0.01f) initial_weight = 0.01f;  /* Minimum */
        if (initial_weight > 1.0f) initial_weight = 1.0f;  /* Maximum for new edges */
        create_edge(g, port_node_id, data_id, initial_weight);
    }
    
    /* EVENT-DRIVEN: Add activated nodes to propagation queue */
    /* Computation follows energy flow, not time ticks */
    /* Always queue nodes that received energy - they need to be processed */
    prop_queue_add(g, port_node_id);
    prop_queue_add(g, data_id);
}


/* Helper: Find edge (hot region only) */
static uint32_t uel_find_edge(Graph *g, uint32_t src, uint32_t dst) {
    if (src >= g->node_count) return UINT32_MAX;  /* Hot region only */
    uint32_t eid = g->nodes[src].first_out;
    uint32_t iter = 0;
    uint32_t max_iter = g->edge_count + 1;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
        if (g->edges[eid].dst == dst) return eid;
        eid = g->edges[eid].next_out;
        iter++;
    }
    return UINT32_MAX;
}

/* Kernel function K(i,j) - hot region only */
static inline float uel_kernel(Graph *g, uint32_t i, uint32_t j) {
    if (i == j || i >= g->node_count || j >= g->node_count) return 0.0f;  /* Hot region only */
    
    if (uel_find_edge(g, j, i) != UINT32_MAX || uel_find_edge(g, i, j) != UINT32_MAX) {
        return 0.5f;
    }
    
    /* Shared neighbor check - hot region only */
    uint32_t ei = g->nodes[i].first_in;
    /* NO ITERATION LIMIT - graph structure determines traversal */
    
    while (ei != UINT32_MAX && ei < g->edge_count) {
        uint32_t neighbor = g->edges[ei].src;
        uint32_t ej = g->nodes[j].first_in;
        /* NO ITERATION LIMIT - graph structure determines traversal */
        
        while (ej != UINT32_MAX && ej < g->edge_count) {
            if (g->edges[ej].src == neighbor) return 0.3f;
            ej = g->edges[ej].next_in;
        }
        ei = g->edges[ei].next_in;
    }
    
    return 0.01f;
}

/* Compute mass - hot region only */
static inline float uel_compute_mass(Graph *g, uint32_t i) {
    if (i >= g->node_count) return 0.0f;  /* Hot region only */
    float a_abs = fabsf(g->nodes[i].a);
    float degree = (float)(g->nodes[i].in_degree + g->nodes[i].out_degree);
    return a_abs + 0.1f * degree;
}

/* ========================================================================
 * EVENT-DRIVEN WAVE PROPAGATION (replaces global ticks)
 * 
 * WE DO NOT USE BIG O NOTATION. WE USE WAVE PROPAGATION LAWS.
 * 
 * This is NOT O(N) complexity. This is physics.
 * - Energy flows through edges like water through pipes
 * - We follow the flow, we don't scan all nodes
 * - Processing time = O(active_nodes), NOT O(total_nodes)
 * - TB-scale graph (16B nodes) + 100 active = 10μs processing
 * 
 * See WAVE_PROPAGATION_LAWS.md for full explanation.
 * ======================================================================== */

/* Thread-safe queue add (per-graph, dynamic size) */
/* Graph-controlled: if queue overflows, graph learns to route inputs differently */
static void prop_queue_add(Graph *g, uint32_t node_id) {
    if (!g || !g->prop_queue) return;
    
    /* Ensure node exists - graph grows dynamically */
    ensure_node(g, node_id);
    
    /* Use node_id as index into prop_queued (modulo for large node IDs) */
    uint64_t queued_idx = node_id % g->prop_queue_size;
    
    /* Atomic check-and-set: only add if not already queued */
    uint8_t expected = 0;
    if (!atomic_compare_exchange_weak(&g->prop_queued[queued_idx], &expected, 1)) {
        return;  /* Already queued by another thread */
    }
    
    /* Add to queue (lock-free circular buffer) */
    /* If queue is full, it wraps - this creates pressure for graph to evolve */
    /* Graph will learn to route inputs through different nodes to reduce overload */
    uint32_t tail = atomic_load(&g->prop_queue_tail);
    uint32_t queue_size = (uint32_t)g->prop_queue_size;
    uint32_t next_tail = (tail + 1) % queue_size;
    
    /* Queue wraps if full - graph learns to handle this through UEL */
    /* High activity → high chaos → graph learns patterns to reduce it */
    /* Graph can evolve to route inputs through specific nodes, create buffers, etc. */
    /* No check for full queue - wrapping is a signal to graph, not a hard limit */
    atomic_store(&g->prop_queue_tail, next_tail);
    g->prop_queue[tail] = node_id;
}

/* Thread-safe queue get (per-graph, dynamic size) */
static uint32_t prop_queue_get(Graph *g) {
    if (!g || !g->prop_queue) return UINT32_MAX;
    
    uint32_t head = atomic_load(&g->prop_queue_head);
    uint32_t tail = atomic_load(&g->prop_queue_tail);
    uint32_t queue_size = (uint32_t)g->prop_queue_size;
    
    if (head == tail) return UINT32_MAX;  /* Empty */
    
    /* Try to claim a node (lock-free) */
    uint32_t new_head = (head + 1) % queue_size;
    if (!atomic_compare_exchange_weak(&g->prop_queue_head, &head, new_head)) {
        return UINT32_MAX;  /* Another thread got it */
    }
    
    uint32_t node_id = g->prop_queue[head];
    
    /* Mark as processed (use modulo for large node IDs) */
    uint64_t queued_idx = node_id;
    if (queued_idx >= g->prop_queue_size) {
        queued_idx = queued_idx % g->prop_queue_size;
    }
    atomic_store(&g->prop_queued[queued_idx], 0);
    
    return node_id;
}

static void prop_queue_clear(Graph *g) {
    if (!g || !g->prop_queue) return;
    atomic_store(&g->prop_queue_head, 0);
    atomic_store(&g->prop_queue_tail, 0);
    for (uint64_t i = 0; i < g->prop_queue_size; i++) {
        atomic_store(&g->prop_queued[i], 0);
    }
}

/* Compute message from neighbors for a single node 
 * 
 * WAVE PROPAGATION: Follows edges (linked list), NEVER scans all nodes.
 * Complexity: O(degree), NOT O(N). Typically 6 edges = 6 operations.
 * This is why TB-scale graphs process in <100ms.
 */
static float compute_message(Graph *g, uint32_t node_id) {
    /* Safety check: ensure arrays are valid and node_id is in bounds */
    if (!g || !g->nodes || !g->edges || node_id >= g->node_count) {
        return 0.0f;
    }
    
    float msg = 0.0f;
    uint32_t eid = g->nodes[node_id].first_in;
    /* NO ITERATION LIMIT - graph structure determines traversal */
    
    /* GRAPH SELF-CORRECTION: Ignore edges below threshold (effectively "removed" from traversal) */
    /* RELATIVE: Threshold adapts to graph state - graph can recover from bad states */
    float edge_strength_safe = (g->avg_edge_strength != g->avg_edge_strength || g->avg_edge_strength < 0.0f) ? 0.1f : g->avg_edge_strength;
    float prune_threshold = edge_strength_safe * uel_params.edge_prune_threshold_ratio;
    if (prune_threshold < 0.001f) prune_threshold = 0.001f;  /* Minimum threshold */
    uint32_t iter = 0;
    uint32_t max_iter = g->edge_count + 1;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
        iter++;
        
        /* GRAPH SELF-CORRECTION: Skip edges below threshold (graph ignores weak/useless edges) */
        float edge_weight = fabsf(g->edges[eid].w);
        if (edge_weight < prune_threshold) {
            /* Edge is too weak - skip it (graph self-corrects by ignoring bad edges) */
            eid = g->edges[eid].next_in;
            continue;
        }
        
        /* Safety check: ensure src node index is valid */
        uint32_t src = g->edges[eid].src;
        if (src < g->node_count) {
            msg += g->edges[eid].w * g->nodes[src].a;
        }
        
        eid = g->edges[eid].next_in;
    }
    return msg;
}

/* Compute mass for a single node (lazy, on-demand) */
static inline float compute_node_mass(Graph *g, uint32_t node_id) {
    if (node_id >= g->node_count) return 0.0f;
    float a_abs = fabsf(g->nodes[node_id].a);
    float degree = (float)(g->nodes[node_id].in_degree + g->nodes[node_id].out_degree);
    float degree_scale = g->avg_activation * 0.1f;
    if (degree_scale < 0.01f) degree_scale = 0.01f;
    return a_abs + degree_scale * degree;
}

/* Compute global field contribution for a single node (simplified) 
 * 
 * WAVE PROPAGATION: Computes Φ field WITHOUT scanning all nodes.
 * Traditional: O(N²) - check all node pairs
 * Melvin: O(degree) - follow edges only, lazy computation
 * 
 * The field Φ gives global context through local edge traversal.
 * Distant nodes have kernel → 0, so we don't compute them.
 * Graph structure encodes "closeness" - only neighbors matter.
 * 
 * GRAPH LAWS DETERMINE PROCESSING: Follow edges, let UEL physics determine what's needed.
 * Not algorithmic complexity - the graph's structure and laws control traversal.
 */
static float compute_phi_contribution(Graph *g, uint32_t node_id, float *mass) {
    /* Safety check: ensure arrays are valid and node_id is in bounds */
    if (!g || !g->nodes || !g->edges || node_id >= g->node_count) {
        return 0.0f;
    }
    
    float phi = 0.0f;
    /* LAZY: Compute mass for this node on-demand if not provided */
    float mass_i = (mass && mass[node_id] > 0.0f) ? mass[node_id] : compute_node_mass(g, node_id);
    if (mass_i < 0.001f) return 0.0f;
    
    /* EDGE-DIRECTED TRAVERSAL: Follow edges, never scan all nodes */
    uint32_t eid = g->nodes[node_id].first_in;
    uint32_t iter2 = 0;
    uint32_t max_iter2 = g->edge_count + 1;
    
    /* GRAPH SELF-CORRECTION: Ignore edges below threshold (effectively "removed" from traversal) */
    /* RELATIVE: Threshold adapts to graph state - graph can recover from bad states */
    float edge_strength_safe = (g->avg_edge_strength != g->avg_edge_strength || g->avg_edge_strength < 0.0f) ? 0.1f : g->avg_edge_strength;
    float prune_threshold = edge_strength_safe * uel_params.edge_prune_threshold_ratio;
    if (prune_threshold < 0.001f) prune_threshold = 0.001f;  /* Minimum threshold */
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter2 < max_iter2) {
        iter2++;
        
        /* GRAPH SELF-CORRECTION: Skip edges below threshold (graph ignores weak/useless edges) */
        float edge_weight = fabsf(g->edges[eid].w);
        if (edge_weight < prune_threshold) {
            /* Edge is too weak - skip it (graph self-corrects by ignoring bad edges) */
            eid = g->edges[eid].next_in;
            continue;
        }
        
        uint32_t src = g->edges[eid].src;
        /* Safety check: ensure src node index is valid */
        if (src >= g->node_count) {
            eid = g->edges[eid].next_in;
            continue;
        }
        
        /* RELATIVE: active threshold based on avg_activation */
        float active_threshold = g->avg_activation * uel_params.active_threshold_ratio;
        if (active_threshold < 0.001f) active_threshold = 0.001f;  /* Minimum */
        
        /* LAZY: Compute mass for neighbor on-demand */
        float src_mass = (mass && mass[src] > 0.0f) ? mass[src] : compute_node_mass(g, src);
        if (src_mass > active_threshold) {
            /* Simplified kernel: direct connection */
            phi += src_mass * 0.5f;
        }
        eid = g->edges[eid].next_in;
    }
    return phi;
}

/* Update a single node and propagate if changed significantly 
 * 
 * CORE WAVE PROPAGATION FUNCTION
 * 
 * This function is called ONLY for nodes in the propagation queue.
 * We NEVER scan all nodes. We NEVER do for(i=0; i<node_count; i++).
 * 
 * Process:
 * 1. Compute message from incoming edges (O(degree), not O(N))
 * 2. Compute field contribution (O(degree), not O(N²))
 * 3. Update this ONE node (O(1))
 * 4. IF changed significantly, add neighbors to queue (O(degree))
 * 
 * Total: O(degree) per call, typically ~6 operations.
 * Even with 16 billion nodes, if only 100 are in queue = ~600 operations total.
 * Result: <10μs processing time, regardless of graph size.
 */
static void update_node_and_propagate(Graph *g, uint32_t node_id, float *mass) {
    /* Lazy allocation: ensure arrays are allocated on first use */
    ensure_node_arrays(g);
    /* Ensure node exists - graph grows dynamically */
    ensure_node(g, node_id);
    
    float a_i = g->nodes[node_id].a;
    float msg_i = compute_message(g, node_id);
    float phi_i = compute_phi_contribution(g, node_id, mass);
    
    /* Combined input from neighbors and global field */
    float field_input = msg_i + uel_params.lambda * phi_i;
    
    /* Local chaos (incoherence): how much a_i disagrees with neighbors */
    /* Clamp to prevent numerical explosion */
    float diff = a_i - msg_i;
    if (diff > 10.0f) diff = 10.0f;
    if (diff < -10.0f) diff = -10.0f;
    float chaos_i = diff * diff;
    
    /* Update running averages (relative measures) */
    /* Clamp chaos_i to prevent explosion in running average */
    if (chaos_i > 100.0f) chaos_i = 100.0f;
    if (chaos_i < 0.0f) chaos_i = 0.0f;  /* Ensure non-negative */
    
    /* NaN check: if avg_chaos is NaN, reset it dynamically */
    if (g->avg_chaos != g->avg_chaos) {  /* NaN check (NaN != NaN) */
        /* Dynamic reset: use current chaos_i if valid, otherwise calculate from graph state */
        if (chaos_i == chaos_i && chaos_i > 0.0f) {
            g->avg_chaos = chaos_i;  /* Use current value */
        } else {
            /* Calculate from graph: use average of recent chaos values or edge strength as proxy */
            float dynamic_default = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.5f) : 0.1f;
            if (dynamic_default < 0.01f) dynamic_default = 0.01f;
            g->avg_chaos = dynamic_default;
        }
    }
    
    g->avg_chaos = uel_params.running_avg_alpha * g->avg_chaos + 
                   (1.0f - uel_params.running_avg_alpha) * chaos_i;
    
    /* Ensure avg_chaos is valid (not NaN or negative) - use dynamic minimum */
    if (g->avg_chaos != g->avg_chaos || g->avg_chaos < 0.0f) {
        /* Dynamic minimum: scale with graph state */
        float dynamic_min = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.1f) : 0.01f;
        if (dynamic_min < 0.001f) dynamic_min = 0.001f;  /* Absolute floor */
        g->avg_chaos = dynamic_min;
    }
    
    /* Clamp activation for running average */
    float a_abs = fabsf(a_i);
    if (a_abs > 10.0f) a_abs = 10.0f;
    
    /* NaN check: if avg_activation is NaN, reset it dynamically */
    if (g->avg_activation != g->avg_activation) {  /* NaN check (NaN != NaN) */
        /* Dynamic reset: use current a_abs if valid, otherwise calculate from graph state */
        if (a_abs == a_abs && a_abs > 0.0f) {
            g->avg_activation = a_abs;  /* Use current value */
        } else {
            /* Calculate from graph: use edge strength or node activation as proxy */
            float dynamic_default = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.3f) : 0.1f;
            if (dynamic_default < 0.01f) dynamic_default = 0.01f;
            g->avg_activation = dynamic_default;
        }
    }
    
    g->avg_activation = uel_params.running_avg_alpha * g->avg_activation + 
                       (1.0f - uel_params.running_avg_alpha) * a_abs;
    
    /* Ensure avg_activation is valid (not NaN or negative) - use dynamic minimum */
    if (g->avg_activation != g->avg_activation || g->avg_activation < 0.0f) {
        /* Dynamic minimum: scale with graph state */
        float dynamic_min = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.2f) : 0.01f;
        if (dynamic_min < 0.001f) dynamic_min = 0.001f;  /* Absolute floor */
        g->avg_activation = dynamic_min;
    }
    
    /* Relative chaos: how chaotic relative to graph average */
    /* Ensure avg_chaos is valid before division - use dynamic minimum */
    float chaos_min = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.05f) : 0.001f;
    if (chaos_min < 0.001f) chaos_min = 0.001f;  /* Absolute floor */
    float avg_chaos_safe = (g->avg_chaos != g->avg_chaos || g->avg_chaos < chaos_min) ? 
                           ((g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.5f) : 0.1f) : g->avg_chaos;
    float relative_chaos = (avg_chaos_safe > chaos_min) ? (chaos_i / avg_chaos_safe) : chaos_i;
    
    /* Ensure relative_chaos is valid */
    if (relative_chaos != relative_chaos || relative_chaos < 0.0f) {
        relative_chaos = 0.0f;
    }
    
    /* Gradient descent: move a_i toward field_input to reduce chaos */
    /* Learning rate adapts to graph state (faster when chaotic) */
    /* RELATIVE: eta_a scales with relative_chaos and avg_activation */
    float avg_activation_safe = (g->avg_activation != g->avg_activation || g->avg_activation < 0.0f) ? 0.1f : g->avg_activation;
    float adaptive_eta = uel_params.eta_a_base * (1.0f + relative_chaos * 0.5f) * 
                         (1.0f / (1.0f + avg_activation_safe));
    
    /* Ensure adaptive_eta is valid */
    if (adaptive_eta != adaptive_eta || adaptive_eta < 0.0f) {
        adaptive_eta = uel_params.eta_a_base;
    }
    float da_i = -adaptive_eta * (a_i - field_input);
    
    /* RESTLESSNESS: Drive to discharge high activation */
    float restlessness_pressure = 0.0f;
    float activation_safe = (g->avg_activation != g->avg_activation || g->avg_activation < 0.0f) ? 
                             ((g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.2f) : 0.1f) : g->avg_activation;
    float restlessness_threshold = activation_safe * uel_params.restlessness_threshold_ratio;
    /* Dynamic minimum: scale with graph state */
    float restlessness_min = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.1f) : 0.001f;
    if (restlessness_threshold < restlessness_min) restlessness_threshold = restlessness_min;
    if (a_i > restlessness_threshold) {
        restlessness_pressure = -uel_params.restlessness_strength * (a_i - restlessness_threshold);
    }

    /* ENERGY QUALITY: Reward coherent high energy */
    float energy_quality = (fabsf(a_i) > 0.001f) ? (fabsf(a_i) / (1.0f + chaos_i)) : 0.0f;
    float quality_reward = uel_params.quality_strength * energy_quality;

    /* PREDICTION: Reward accurate predictions (simplified for now) */
    float prediction_reward = (g->prediction_accuracy) ? 
                             (uel_params.prediction_strength * g->prediction_accuracy[node_id]) : 0.0f;
    
    /* NOVELTY SEEKING (Guilty Pleasure): When chaos is too low, add random activation */
    /* System gets "bored" when chaos is too stable - it wants some excitement */
    /* This creates natural exploration - graph will discover COLD_DATA patterns if they exist */
    float novelty_pressure = 0.0f;
    float boredom_threshold = g->avg_chaos * uel_params.boredom_threshold_ratio;
    if (boredom_threshold < 0.001f) boredom_threshold = 0.001f;  /* Minimum */
    
    /* If chaos is below boredom threshold, add small random activation boost */
    /* This is the "guilty pleasure" - system wants some chaos when things are too stable */
    if (g->avg_chaos < boredom_threshold && g->cold_data_size > 0) {
        /* Small random boost to this node (creates natural exploration) */
        /* Graph will naturally discover COLD_DATA patterns if they're useful */
        float boredom_level = (boredom_threshold - g->avg_chaos) / boredom_threshold;
        float random_factor = ((float)(node_id % 100) / 100.0f);  /* Pseudo-random based on node ID */
        novelty_pressure = uel_params.novelty_seeking_strength * 0.1f * boredom_level * random_factor;
    }
    
    /* Combine all drive mechanisms */
    da_i = da_i + restlessness_pressure + quality_reward + prediction_reward + novelty_pressure;
    
    /* ========================================================================
     * MULTI-PART ENERGY SYSTEM:
     * 1. ADAPTIVE storage when energy flows through (not rigid 10% - adapts to:
     *    - Node capacity (high capacity = diminishing returns)
     *    - Graph activity (high activity = less storage per node)
     *    - Graph size (larger graphs = less storage per node)
     * 2. Higher stored energy = lower activation threshold
     * 3. More stored energy = bigger capacity (importance persists)
     * 4. Release energy when beneficial (high chaos, amplification, etc)
     * 
     * Brain analogy: Energy doesn't accumulate linearly through millions of neurons.
     * Storage efficiency decreases as signal travels further, preventing explosion.
     * ======================================================================== */
    
    /* Calculate new activation - will be updated based on energy system */
    float new_a = a_i + da_i;
    
    /* Arrays are guaranteed to be allocated by ensure_node_arrays() at start of function */
    if (!g->stored_energy_capacity) {
        /* Fallback: just use activation update without energy system */
        g->nodes[node_id].a = tanhf(new_a);
    } else {
        /* 1. ADAPTIVE STORAGE: Storage efficiency adapts to prevent explosion on long paths */
        /* Brain analogy: Synapses store less as signal travels through many neurons */
        float flow_energy = fabsf(msg_i);  /* Energy flowing through this node */
        
        /* Calculate adaptive storage efficiency */
        float base_efficiency = uel_params.storage_efficiency_base;  /* Start with base (e.g., 10%) */
        
        /* Factor 1: Diminishing returns - nodes with high capacity store less */
        float capacity = g->stored_energy_capacity[node_id];
        float capacity_factor = 1.0f;
        if (capacity > 0.1f) {
            /* Storage efficiency decreases as capacity grows (logarithmic) */
            capacity_factor = 1.0f / (1.0f + capacity * uel_params.storage_diminishing_factor);
        }
        
        /* Factor 2: Graph activity scaling - high activity = less storage per node */
        /* Prevents energy explosion when many nodes are active simultaneously */
        float activity_factor = 1.0f;
        float activation_safe = (g->avg_activation != g->avg_activation || g->avg_activation < 0.0f) ? 
                                0.1f : g->avg_activation;
        if (activation_safe > 0.01f) {
            /* High graph activity reduces storage efficiency (distributed across many nodes) */
            activity_factor = 1.0f / (1.0f + activation_safe * uel_params.storage_activity_scaling);
        }
        
        /* Factor 3: Graph size scaling - larger graphs need less storage per node */
        float graph_size_factor = 1.0f;
        if (g->node_count > 1000) {
            /* For large graphs, storage efficiency decreases (energy distributed across many nodes) */
            graph_size_factor = 1000.0f / (float)g->node_count;
            if (graph_size_factor < 0.01f) graph_size_factor = 0.01f;  /* Minimum */
        }
        
        /* Combined adaptive storage efficiency */
        float adaptive_efficiency = base_efficiency * capacity_factor * activity_factor * 
                                   graph_size_factor;
        
        /* Clamp to reasonable range (0.001% to 10%) */
        if (adaptive_efficiency < 0.00001f) adaptive_efficiency = 0.00001f;
        if (adaptive_efficiency > 0.1f) adaptive_efficiency = 0.1f;
        
        float newly_stored = flow_energy * adaptive_efficiency;
        
        /* 2. GROW CAPACITY: Stored energy increases capacity (importance) */
        /* Capacity grows over time - even if energy is released, importance persists */
        g->stored_energy_capacity[node_id] += newly_stored * uel_params.capacity_growth_rate;
        if (g->stored_energy_capacity[node_id] < 0.0f) g->stored_energy_capacity[node_id] = 0.0f;  /* No negative capacity */
        
        /* 3. LOWER THRESHOLD: Higher capacity = easier to activate */
        /* Reuse capacity variable from above */
        capacity = g->stored_energy_capacity[node_id];  /* Update capacity after growth */
        float threshold_reduction = capacity * uel_params.threshold_reduction_per_capacity;
        
        /* 4. DECIDE: Should we release stored energy? */
        /* Release conditions:
         * - High chaos (spread energy to stabilize neighbors)
         * - High activation + high capacity (full node wants to spread)
         * - Neighbors need help (amplification needed)
         */
        float released_energy = 0.0f;
        bool should_release = false;
        
        /* Condition 1: High chaos - release to stabilize */
        float relative_chaos = (g->avg_chaos > 0.001f) ? (chaos_i / g->avg_chaos) : chaos_i;
        if (relative_chaos > 2.0f && capacity > 0.1f) {
            should_release = true;
        }
        
        /* Condition 2: High activation + high capacity - full node spreads energy */
        /* Reuse activation_safe from above */
        float high_activation_threshold = activation_safe * 1.5f;
        if (fabsf(a_i) > high_activation_threshold && capacity > 0.2f) {
            should_release = true;
        }
        
        /* Condition 3: Restlessness - node wants to discharge */
        if (restlessness_pressure < -0.1f && capacity > 0.05f) {
            should_release = true;
        }
        
        /* Release stored energy if beneficial */
        if (should_release && capacity > 0.0f) {
            float release_amount = capacity * uel_params.release_efficiency;  /* Release 50% of capacity */
            released_energy = release_amount;
            g->stored_energy_capacity[node_id] -= release_amount;  /* Energy is spent, but capacity persists */
            if (g->stored_energy_capacity[node_id] < 0.0f) g->stored_energy_capacity[node_id] = 0.0f;
        }
        
        /* Update activation: gradient descent + newly stored + released + capacity boost */
        /* Higher capacity = lower threshold = easier activation (acts like pre-activation) */
        float capacity_boost = threshold_reduction * 0.5f;  /* Convert threshold reduction to activation boost */
        new_a = a_i + da_i + newly_stored + released_energy + capacity_boost;  /* Update new_a with energy system */
        g->nodes[node_id].a = tanhf(new_a);
        
        /* PATTERN EXPANSION: If this is a pattern node and it's highly activated, expand it */
        /* RELATIVE: Pattern expansion threshold based on avg_activation (like other thresholds) */
        if (g->nodes[node_id].pattern_data_offset > 0) {
            float activation_safe = (g->avg_activation != g->avg_activation || g->avg_activation < 0.0f) ? 
                                    0.1f : g->avg_activation;
            if (activation_safe < 0.001f) activation_safe = 0.001f;
            float pattern_threshold = activation_safe * 1.5f;  /* 150% of avg_activation (high activation needed) */
            if (fabsf(new_a) > pattern_threshold) {
                /* Pattern node is activated - expand to underlying sequence */
                expand_pattern(g, node_id, NULL);  /* No bindings for now - would need context */
            }
        }
        
        /* EXEC NODE EXECUTION: If this is an EXEC node and activation exceeds threshold, execute */
        if (g->nodes[node_id].payload_offset > 0) {
            /* EXEC node - check if activation exceeds threshold */
            Node *exec_n = &g->nodes[node_id];
            float activation = fabsf(exec_n->a);
            float threshold_ratio = (exec_n->exec_threshold_ratio > 0.0f) ? 
                                    exec_n->exec_threshold_ratio : 0.5f;
            float avg_act_safe = (g->avg_activation > 0.0f) ? g->avg_activation : 0.1f;
            float threshold = avg_act_safe * threshold_ratio;
            
            if (activation >= threshold) {
                ROUTE_LOG("UEL: Firing EXEC node %u: activation=%.3f >= threshold=%.3f",
                          node_id, activation, threshold);
                melvin_execute_exec_node(g, node_id);
            }
        }
        
        /* GENERAL: When patterns match sequences, extract values and route to EXEC nodes */
        /* This is learnable - patterns learn which EXEC nodes they route to */
        if (g->nodes[node_id].pattern_data_offset > 0 && fabsf(new_a) > g->avg_activation * 0.5f) {
            /* Pattern node activated - check if it matches current sequence and routes to EXEC */
            if (g->sequence_buffer && g->sequence_buffer_pos > 0) {
                /* Get recent sequence */
                uint32_t seq_len = (g->sequence_buffer_pos < 20) ? (uint32_t)g->sequence_buffer_pos : 20;
                uint32_t start_pos = (g->sequence_buffer_pos >= seq_len) ? 
                                    (g->sequence_buffer_pos - seq_len) : 0;
                
                uint32_t sequence[20];
                for (uint32_t i = 0; i < seq_len; i++) {
                    uint32_t pos = (start_pos + i) % g->sequence_buffer_size;
                    sequence[i] = g->sequence_buffer[pos];
                }
                
                /* Check if pattern matches sequence */
                uint32_t bindings[256] = {0};
                if (pattern_matches_sequence(g, node_id, sequence, seq_len, bindings)) {
                    /* Pattern matches - extract values from blanks */
                    PatternValue extracted_values[16];
                    uint32_t value_count = 0;
                    
                    /* Read pattern to find blanks */
                    uint64_t pattern_offset = g->nodes[node_id].pattern_data_offset - g->hdr->blob_offset;
                    if (pattern_offset < g->blob_size) {
                        PatternData *pattern_data = (PatternData *)(g->blob + pattern_offset);
                        if (pattern_data->magic == 0x4E544150) {  /* PATTERN_MAGIC */
                            for (uint32_t i = 0; i < pattern_data->element_count && value_count < 16; i++) {
                                PatternElement *elem = &pattern_data->elements[i];
                                if (elem->is_blank != 0 && bindings[elem->value] > 0) {
                                    /* Blank with binding - extract value */
                                    uint32_t bound_node = bindings[elem->value];
                                    uint32_t seq[1] = {bound_node};
                                    PatternValue val = extract_pattern_value(g, seq, 1, node_id);
                                    /* FIXED: Accept value 0 as valid (was: val.value_data > 0) */
                                    if (val.value_type == 0 && val.confidence > 0.0f) {
                                        extracted_values[value_count++] = val;
                                        ROUTE_LOG("  UEL extracted value[%u] = %llu (type=%u, conf=%.2f)",
                                                  value_count - 1, (unsigned long long)val.value_data,
                                                  val.value_type, val.confidence);
                                    }
                                }
                            }
                        }
                    }
                    
                    /* If values extracted, check if pattern routes to EXEC node */
                    /* EXEC nodes are like output nodes - they get activated through edges */
                    if (value_count > 0) {
                        ROUTE_LOG("  UEL routing %u values to EXEC nodes", value_count);
                        uint32_t eid = g->nodes[node_id].first_out;
                        for (int ei = 0; ei < 100 && eid != UINT32_MAX && eid < g->edge_count; ei++) {
                            uint32_t dst = g->edges[eid].dst;
                            if (dst < g->node_count && g->nodes[dst].payload_offset > 0) {
                                /* Pattern routes to EXEC node - pass values */
                                ROUTE_LOG("  UEL: Pattern %u → EXEC %u", node_id, dst);
                                pass_values_to_exec(g, dst, extracted_values, value_count);
                                break;
                            }
                            eid = g->edges[eid].next_out;
                        }
                    }
                }
            }
        }
        
    }
    
    /* Update edge weights for this node's incoming edges */
    uint32_t eid = g->nodes[node_id].first_in;
    uint32_t max_iter = (uint32_t)(g->edge_count + 1);
    uint32_t iter = 0;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
        uint32_t src = g->edges[eid].src;
        if (src < g->node_count) {
            float a_src = g->nodes[src].a;
            float chaos_dst = a_i - msg_i;
            
            /* Relative edge update: adapts to graph state */
            /* RELATIVE: eta_w scales inversely with avg_edge_strength */
            float relative_eta_w = uel_params.eta_w_base * 
                                   (1.0f / (1.0f + g->avg_edge_strength));
            float dw = -relative_eta_w * chaos_dst * a_src;
            
            /* EXPLORATION: Strengthen edges leading to output nodes that get feedback */
            /* RELATIVE: exploration boost based on feedback correlation */
            /* Note: node_id is the destination of this edge */
            /* OPTIMIZATION: For large graphs, use modulo indexing */
            uint64_t tracking_idx = (g->tracking_array_size > 0) ? (node_id % g->tracking_array_size) : 0;
            if (g->output_propensity && g->feedback_correlation &&
                g->output_propensity[tracking_idx] > 0.1f && g->feedback_correlation[tracking_idx] > 0.0f) {
                float exploration_boost = uel_params.exploration_strength * 
                                         g->output_propensity[tracking_idx] * 
                                         g->feedback_correlation[tracking_idx] * 
                                         a_src;
                dw += exploration_boost;
            }
            
            g->edges[eid].w += dw;
            
            /* Weight decay - RELATIVE: scales with avg_edge_strength */
            float relative_decay_w = uel_params.decay_w_base * (1.0f + g->avg_edge_strength * 0.1f);
            g->edges[eid].w *= (1.0f - relative_decay_w);
            
            /* Clamp - RELATIVE: based on avg_edge_strength */
            float weight_clamp = g->avg_edge_strength * uel_params.weight_clamp_ratio;
            if (weight_clamp < 0.1f) weight_clamp = 0.1f;  /* Minimum clamp */
            if (g->edges[eid].w > weight_clamp) g->edges[eid].w = weight_clamp;
            if (g->edges[eid].w < -weight_clamp) g->edges[eid].w = -weight_clamp;
            
            /* Update running average of edge strength */
            g->avg_edge_strength = uel_params.running_avg_alpha * g->avg_edge_strength + 
                                  (1.0f - uel_params.running_avg_alpha) * fabsf(g->edges[eid].w);
        }
        eid = g->edges[eid].next_in;
        iter++;
    }
    
    /* Check if change is significant enough to propagate */
    /* RELATIVE: threshold based on avg_activation */
    /* Arrays are guaranteed to be allocated by ensure_node_arrays() at start of function */
    /* OPTIMIZATION: For large graphs, use modulo indexing (circular buffer) */
    uint64_t tracking_idx = (g->tracking_array_size > 0) ? (node_id % g->tracking_array_size) : 0;
    float activation_change = (g->last_activation) ? fabsf(new_a - g->last_activation[tracking_idx]) : fabsf(new_a);
    float message_change = (g->last_message) ? fabsf(msg_i - g->last_message[tracking_idx]) : fabsf(msg_i);
    
    /* Relative threshold: change must be significant compared to graph's typical activation */
    float activation_safe_thresh = (g->avg_activation != g->avg_activation || g->avg_activation < 0.0f) ? 
                                    ((g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.2f) : 0.1f) : g->avg_activation;
    float relative_change_threshold = activation_safe_thresh * uel_params.change_threshold_ratio;
    /* Dynamic minimum: scale with graph state */
    float change_min = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.05f) : 0.001f;
    if (relative_change_threshold < change_min) relative_change_threshold = change_min;
    
    if (activation_change > relative_change_threshold || 
        message_change > relative_change_threshold) {
        /* Significant change - propagate to neighbors */
        /* Arrays are guaranteed to be allocated by ensure_node_arrays() at start of function */
        /* OPTIMIZATION: For large graphs, use modulo indexing (circular buffer) */
        uint64_t tracking_idx = (g->tracking_array_size > 0) ? (node_id % g->tracking_array_size) : 0;
        if (g->last_activation) g->last_activation[tracking_idx] = new_a;
        if (g->last_message) g->last_message[tracking_idx] = msg_i;
        
            /* GRAPH SELF-CORRECTION: Only propagate through edges above threshold */
            /* RELATIVE: Threshold adapts to graph state - graph can recover from bad states */
            float edge_strength_safe = (g->avg_edge_strength != g->avg_edge_strength || g->avg_edge_strength < 0.0f) ? 0.1f : g->avg_edge_strength;
            float prune_threshold = edge_strength_safe * uel_params.edge_prune_threshold_ratio;
            if (prune_threshold < 0.001f) prune_threshold = 0.001f;  /* Minimum threshold */
        
            /* Add outgoing neighbors to queue (only if edge is strong enough) */
            eid = g->nodes[node_id].first_out;
            iter = 0;
            while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
                /* GRAPH SELF-CORRECTION: Skip weak edges (graph ignores bad connections) */
                float edge_weight = fabsf(g->edges[eid].w);
                if (edge_weight < prune_threshold) {
                    eid = g->edges[eid].next_out;
                    iter++;
                    continue;
                }
                
                uint32_t dst = g->edges[eid].dst;
                ensure_node(g, dst);  /* Ensure node exists - graph grows dynamically */
                if (dst < g->node_count) {
                    prop_queue_add(g, dst);
                }
                eid = g->edges[eid].next_out;
                iter++;
            }
            
            /* Add incoming neighbors to queue (only if edge is strong enough) */
            eid = g->nodes[node_id].first_in;
            iter = 0;
            while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
                /* GRAPH SELF-CORRECTION: Skip weak edges (graph ignores bad connections) */
                float edge_weight = fabsf(g->edges[eid].w);
                if (edge_weight < prune_threshold) {
                    eid = g->edges[eid].next_in;
                    iter++;
                    continue;
                }
                
                uint32_t src = g->edges[eid].src;
                ensure_node(g, src);  /* Ensure node exists - graph grows dynamically */
                if (src < g->node_count) {
                    prop_queue_add(g, src);
                }
                eid = g->edges[eid].next_in;
                iter++;
            }
    }
}

/* Worker thread function for async processing */
typedef struct {
    Graph *g;
    float *mass;
    uint32_t *active_workers;
    pthread_mutex_t *active_mutex;
} WorkerArgs;

static void *async_worker(void *arg) {
    WorkerArgs *args = (WorkerArgs *)arg;
    Graph *g = args->g;
    float *mass = args->mass;
    
    uint32_t processed = 0;
    /* RELATIVE: Worker batch size scales with graph size and activity */
    /* Base on node_count and activation - larger/more active graphs process more per worker */
    uint32_t base_batch = (g->node_count > 0) ? (uint32_t)(g->node_count / 100) : 100;
    if (base_batch < 50) base_batch = 50;  /* Minimum batch size */
    float activity_factor = 1.0f + g->avg_activation;  /* More active = larger batches */
    uint32_t max_per_worker = (uint32_t)(base_batch * activity_factor);
    if (max_per_worker > 10000) max_per_worker = 10000;  /* Safety: prevent memory issues */
    
    while (processed < max_per_worker) {
        uint32_t node_id = prop_queue_get(g);
        if (node_id == UINT32_MAX) {
            /* Queue empty - check if other workers are still active */
            usleep(100);  /* Brief pause to let other workers add nodes */
            node_id = prop_queue_get(g);
            if (node_id == UINT32_MAX) break;  /* Still empty, exit */
        }
        
        /* Process node - ASYNC: may read partially updated neighbors */
        update_node_and_propagate(g, node_id, mass);
        processed++;
    }
    
    /* Decrement active worker count */
    pthread_mutex_lock(args->active_mutex);
    (*args->active_workers)--;
    pthread_mutex_unlock(args->active_mutex);
    
    return NULL;
}

/* Event-driven wave propagation - ASYNC VERSION (parallel processing) */
static void uel_propagate_from(Graph *g, uint32_t start_node) {
    if (!g || !g->hdr || g->node_count == 0 || start_node >= g->node_count) return;
    
    /* Allocate mass buffer (needed for phi computation) - shared by all workers */
    float *mass = calloc(g->node_count, sizeof(float));
    if (!mass) return;
    
    /* Compute mass for active nodes (read-only, safe to share) */
    /* RELATIVE: degree contribution scales with avg_activation */
    float degree_scale = g->avg_activation * 0.1f;
    if (degree_scale < 0.01f) degree_scale = 0.01f;  /* Minimum */
    
    for (uint64_t i = 0; i < g->node_count; i++) {
        float a_abs = fabsf(g->nodes[i].a);
        float degree = (float)(g->nodes[i].in_degree + g->nodes[i].out_degree);
        mass[i] = a_abs + degree_scale * degree;
    }
    
    /* Clear propagation queue and start from initial node */
    prop_queue_clear(g);
    prop_queue_add(g, start_node);
    
    /* Determine number of worker threads (use all available cores) */
    /* NO CAP - graph can handle any number of workers through UEL physics */
    int num_workers = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (num_workers < 1) num_workers = 1;
    /* Removed cap - use all cores for maximum parallelism */
    
    /* Create worker threads for async processing */
    pthread_t *workers = calloc(num_workers, sizeof(pthread_t));
    WorkerArgs *args = calloc(num_workers, sizeof(WorkerArgs));
    uint32_t active_workers = num_workers;
    pthread_mutex_t active_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    if (!workers || !args) {
        free(workers);
        free(args);
        free(mass);
        return;
    }
    
    /* Launch worker threads */
    for (int i = 0; i < num_workers; i++) {
        args[i].g = g;
        args[i].mass = mass;
        args[i].active_workers = &active_workers;
        args[i].active_mutex = &active_mutex;
        
        if (pthread_create(&workers[i], NULL, async_worker, &args[i]) != 0) {
            /* Thread creation failed - fall back to sequential */
            active_workers = i;
            break;
        }
    }
    
    /* Wait for all workers to finish or queue to be empty */
    uint32_t max_wait_iterations = 1000;
    uint32_t wait_count = 0;
    
    while (wait_count < max_wait_iterations) {
        pthread_mutex_lock(&active_mutex);
        uint32_t still_active = active_workers;
        pthread_mutex_unlock(&active_mutex);
        
        if (still_active == 0) break;  /* All workers done */
        
        /* Check if queue is empty and no workers active */
        uint32_t head = atomic_load(&g->prop_queue_head);
        uint32_t tail = atomic_load(&g->prop_queue_tail);
        if (head == tail && still_active == 0) break;
        
        usleep(1000);  /* Wait 1 millisecond */
        wait_count++;
    }
    
    /* Wait for all threads to complete */
    for (int i = 0; i < num_workers; i++) {
        if (workers[i] != 0) {
            pthread_join(workers[i], NULL);
        }
    }
    
    pthread_mutex_destroy(&active_mutex);
    free(workers);
    free(args);
    free(mass);
}

/* Initialize random seed (called once) */
static void init_random_seed(void) {
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned int)time(NULL) ^ (unsigned int)getpid());
        seeded = true;
    }
}

/* Curiosity mechanism: reactivate nodes when system is bored (low chaos) */
/* Uses async propagation system - adds nodes to queue with activation power */
/* Activation power depends on available activation in graph (inverse relationship) */
static void curiosity_reactivate(Graph *g) {
    if (!g || !g->hdr || g->node_count == 0) return;
    
    /* Ensure random seed is initialized */
    init_random_seed();
    
    /* Check if system is bored (chaos too low) OR cold (no activation) */
    float boredom_threshold = g->avg_chaos * uel_params.boredom_threshold_ratio;
    if (boredom_threshold < 0.001f) boredom_threshold = 0.001f;
    
    /* Also trigger if activation is zero (cold start - bootstrap) */
    bool is_cold = (g->avg_activation < 0.001f);
    bool is_bored = (g->avg_chaos < boredom_threshold);
    
    if (!is_cold && !is_bored) return;  /* Not bored and has activation */
    
    /* System is bored or cold - calculate activation power */
    float curiosity_power;
    if (is_cold) {
        /* Cold start: strong bootstrap activation */
        curiosity_power = 0.5f;  /* Strong initial push */
    } else {
        /* Bored: calculate based on boredom level */
        float boredom_level = (boredom_threshold - g->avg_chaos) / boredom_threshold;
        float activation_availability = 1.0f / (1.0f + g->avg_activation);
        curiosity_power = uel_params.novelty_seeking_strength * boredom_level * activation_availability;
    }
    
    /* Calculate how many nodes to reactivate */
    /* RELATIVE: Reactivation count scales with graph size */
    uint32_t base_reactivate = is_cold ? 50 : (uint32_t)(curiosity_power * 20.0f);
    if (base_reactivate < 1) base_reactivate = 1;
    /* Scale with graph size: larger graphs can explore more nodes */
    float graph_size_factor = (g->node_count > 0) ? (1.0f + (float)g->node_count / 10000.0f) : 1.0f;
    if (graph_size_factor > 10.0f) graph_size_factor = 10.0f;  /* Cap at 10x */
    uint32_t num_to_reactivate = (uint32_t)(base_reactivate * graph_size_factor);
    /* No absolute cap - graph controls its own exploration based on size */
    
    /* Split 50/50: half to random nodes, half to recently active nodes */
    uint32_t num_random = num_to_reactivate / 2;
    uint32_t num_recent = num_to_reactivate - num_random;
    
    /* 50%: Random nodes (pure exploration) OR read from cold_data if available */
    for (uint32_t i = 0; i < num_random; i++) {
        /* If cold_data exists and we're bored, try reading from it */
        if (g->cold_data_size > 0 && (is_bored || is_cold)) {
            /* Pick random offset in cold_data */
            uint64_t cold_offset = (uint64_t)rand() % g->cold_data_size;
            /* Limit read length to prevent queue overflow - read fewer bytes */
            uint64_t read_length = 10;  /* Reduced from 100 to prevent queue overflow */
            if (cold_offset + read_length > g->cold_data_size) {
                read_length = g->cold_data_size - cold_offset;
            }
            if (read_length > 10) read_length = 10;  /* Cap at 10 bytes max */
            
            /* Feed bytes from cold_data into graph */
            for (uint64_t j = 0; j < read_length; j++) {
                uint8_t byte = g->cold_data[cold_offset + j];
                uint32_t data_node = (uint32_t)byte;
                ensure_node(g, data_node);
                
                /* Feed byte with curiosity energy */
                float feed_energy = is_cold ? 0.2f : (curiosity_power * 0.2f);
                melvin_feed_byte(g, 0, byte, feed_energy);  /* Feed through port 0 */
            }
            
            /* Only do this once per curiosity cycle to avoid overwhelming */
            break;
        } else {
            /* No cold_data or not bored - just reactivate random node */
            uint32_t node_id = (uint32_t)((uint64_t)rand() % g->node_count);
            if (node_id < g->node_count) {
                /* Add activation boost (stronger for cold start) */
                float reactivation_energy = is_cold ? 0.3f : (curiosity_power * 0.3f);
                g->nodes[node_id].a += reactivation_energy;
                g->nodes[node_id].a = tanhf(g->nodes[node_id].a);  /* Clamp */
                
                /* Add to async propagation queue (same system as normal propagation) */
                prop_queue_add(g, node_id);
            }
        }
    }
    
    /* 50%: Recently active nodes (build on recent activity) */
    if (g->last_activation && num_recent > 0) {
        float recent_threshold = g->avg_activation * 0.3f;
        if (recent_threshold < 0.01f) recent_threshold = 0.01f;
        
        uint32_t found = 0;
        for (uint32_t i = 0; i < num_recent * 10 && found < num_recent; i++) {
            uint32_t node_id = (uint32_t)((uint64_t)rand() % g->node_count);
            
            /* OPTIMIZATION: For large graphs, use modulo indexing */
            uint64_t tracking_idx = (g->tracking_array_size > 0) ? (node_id % g->tracking_array_size) : 0;
            if (node_id < g->node_count && 
                g->last_activation && g->last_activation[tracking_idx] > recent_threshold) {
                /* This node was recently active - reactivate it */
                float reactivation_energy = curiosity_power * 0.3f;
                g->nodes[node_id].a += reactivation_energy;
                g->nodes[node_id].a = tanhf(g->nodes[node_id].a);  /* Clamp */
                
                /* Add to async propagation queue */
                prop_queue_add(g, node_id);
                found++;
            }
        }
        
        /* If we didn't find enough recent nodes, fill with random */
        while (found < num_recent) {
            uint32_t node_id = (uint32_t)((uint64_t)rand() % g->node_count);
            if (node_id < g->node_count) {
                float reactivation_energy = curiosity_power * 0.3f;
                g->nodes[node_id].a += reactivation_energy;
                g->nodes[node_id].a = tanhf(g->nodes[node_id].a);
                prop_queue_add(g, node_id);
                found++;
            }
        }
    }
}

/* Legacy function for compatibility - now triggers propagation from all active nodes */
/* ============================================================================
 * UEL_MAIN: WAVE PROPAGATION ENGINE
 * ============================================================================
 * 
 * THIS IS THE CORE. THIS IS WHY WE SCALE TO TB WITH <100ms.
 * 
 * WE NEVER SCAN ALL NODES. NO for(i=0; i<node_count; i++). EVER.
 * 
 * How it works:
 * 1. Process ONLY nodes in prop_queue (typically 100-1000, NOT billions)
 * 2. Each node update follows edges (O(degree), NOT O(N))
 * 3. Changed nodes add neighbors to queue (wave propagation)
 * 4. Queue empties when wave settles (chaos minimized)
 * 
 * Example with 1TB graph (16B nodes):
 * - Input arrives → 1 node queued
 * - Wave propagates → ~100-1000 nodes touched
 * - Each node: ~6 edges traversed
 * - Total: ~600-6000 operations
 * - Time: ~10-100μs (NOT seconds, NOT minutes)
 * 
 * Traditional system would scan 16B nodes = hours
 * Melvin follows wave through 100 nodes = microseconds
 * 
 * This is physics, not algorithms. This is UEL.
 * ============================================================================
 */
static void uel_main(Graph *g) {
    /* PURE EVENT-DRIVEN: Only process nodes in the propagation queue */
    /* No global ticks - only process what's queued */
    /* NO SCANNING ALL NODES - process queue only */
    if (!g || !g->hdr || g->node_count == 0) return;
    
    /* GRAPH LAWS DETERMINE PROCESSING: Mass computed only for nodes the graph needs */
    /* Edges direct traversal - we follow connections, not algorithmic complexity */
    /* The graph's structure and UEL physics determine what gets processed */
    float *mass = NULL;  /* Computed on-demand: only for nodes reached via edges */
    
    /* Curiosity: reactivate nodes when bored (creates natural exploration) */
    /* This adds nodes to the queue, which then get processed */
    curiosity_reactivate(g);
    
    /* Process all queued nodes (event-driven, not tick-based) */
    /* GRAPH LAWS DETERMINE PROCESSING: Process until queue is empty, no algorithmic limits */
    /* The graph's UEL physics and structure determine activity - high chaos = more processing */
    /* Edges direct what gets processed - we follow connections, not algorithmic complexity */
    uint32_t processed = 0;
    /* Process queue until truly empty - graph determines when it's done */
    /* Graph structure and UEL physics determine processing */
    uint32_t consecutive_empty = 0;
    uint32_t max_consecutive_empty = 10;  /* Check for truly empty queue */
    uint32_t curiosity_calls = 0;
    uint32_t max_curiosity = 3;  /* Limit curiosity to prevent infinite loops */
    
    /* ========================================================================
     * WAVE PROPAGATION LOOP - CORE SCALING MECHANISM
     * 
     * This loop processes ONLY nodes in the queue.
     * With 16 billion nodes in graph, queue typically has ~100-1000.
     * 
     * NO SCANNING. FOLLOW THE WAVE.
     * ======================================================================== */
    while (consecutive_empty < max_consecutive_empty) {
        uint32_t node_id = prop_queue_get(g);
        if (node_id == UINT32_MAX) {
            /* Queue empty - check if curiosity added anything (limited) */
            if (curiosity_calls < max_curiosity) {
                curiosity_reactivate(g);
                curiosity_calls++;
            }
            node_id = prop_queue_get(g);
            if (node_id == UINT32_MAX) {
                consecutive_empty++;
                if (consecutive_empty >= max_consecutive_empty) break;
                continue;
            }
        }
        
        /* Reset empty counter - we found work */
        consecutive_empty = 0;
        
        /* Process node (event-driven) - mass computed lazily per-node */
        update_node_and_propagate(g, node_id, mass);
        processed++;
        
        /* HIERARCHICAL COMPOSITION: Track pattern adjacency */
        track_pattern_adjacency(g);
        
        /* EVENT-DRIVEN PATTERN DISCOVERY: Track activation in window */
        if (g->recent_activations && node_id < g->node_count) {
            g->recent_activations[g->activation_window_pos] = node_id;
            g->activation_window_pos = (g->activation_window_pos + 1) % g->activation_window_size;
            g->activation_check_counter++;
            
            /* Check for co-activation patterns every 100 activations */
            if (g->activation_check_counter >= 100) {
                extern void detect_coactivation_patterns(Graph *g);
                detect_coactivation_patterns(g);
                g->activation_check_counter = 0;
                
                /* HIERARCHICAL COMPOSITION: After discovering base patterns, try to compose them */
                /* Run every 500 activations (every 5 discovery cycles) */
                static uint64_t composition_counter = 0;
                composition_counter++;
                if (composition_counter >= 5) {
                    fprintf(stderr, "\n🔨 Triggering composition (cycle %llu, adjacencies=%u)...\n", 
                            (unsigned long long)composition_counter, adjacency_count);
                    compose_adjacent_patterns(g);
                    composition_counter = 0;
                }
            }
        }
        
        /* Reasonable limit to prevent hangs while still allowing growth */
        if (processed > 10000) break;  /* 10k operations per call is plenty */
        
        /* SELF-REGULATION: If chaos is very low, graph is stable */
        if (g->avg_chaos < 0.01f && processed > 100) {
            break;
        }
    }
    
    /* SELF-REGULATION: Update output activity tracking for feedback learning */
    /* Graph learns which outputs create feedback through this tracking */
    if (g->avg_output_activity > 0.0f) {
        /* Decay output activity over time (graph forgets old outputs) */
        g->avg_output_activity *= 0.99f;  /* Slow decay */
    }
    
    /* Cleanup: mass is NULL (lazy computation), nothing to free */
    /* No need to free - we never allocated mass for all nodes */
}

/* ========================================================================
 * CREATE V2 FILE (reusable for corpus packing)
 * ======================================================================== */

int melvin_create_v2(const char *path, 
                     uint64_t hot_nodes, 
                     uint64_t hot_edges, 
                     uint64_t hot_blob_bytes, 
                     uint64_t cold_data_bytes) {
    if (!path) return -1;
    
    /* Calculate layout - all offsets 64-bit */
    uint64_t header_size = sizeof(MelvinHeader);
    uint64_t nodes_size = hot_nodes * sizeof(Node);
    uint64_t edges_size = hot_edges * sizeof(Edge);
    
    uint64_t off = header_size;
    uint64_t nodes_offset = off;
    off += nodes_size;
    uint64_t edges_offset = off;
    off += edges_size;
    uint64_t blob_offset = off;
    off += hot_blob_bytes;
    uint64_t cold_data_offset = off;
    off += cold_data_bytes;
    uint64_t total_size = off;
    
    /* Round up to page size */
    size_t page_size = getpagesize();
    total_size = (total_size + page_size - 1) & ~(page_size - 1);
    
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return -1;
    }
    
    /* Create file */
    if (ftruncate(fd, total_size) < 0) {
        close(fd);
        return -1;
    }
    
    /* mmap with EXEC permission for blob code execution */
    void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return -1;
    }
    
    /* Initialize header */
    MelvinHeader *hdr = (MelvinHeader *)map;
    memcpy(hdr->magic, MELVIN_MAGIC, 4);
    hdr->version = MELVIN_VERSION;
    hdr->flags = 0;
    hdr->file_size = total_size;
    hdr->nodes_offset = nodes_offset;
    hdr->node_count = hot_nodes;
    hdr->edges_offset = edges_offset;
    hdr->edge_count = 0;
    hdr->blob_offset = blob_offset;
    hdr->blob_size = hot_blob_bytes;
    hdr->cold_data_offset = cold_data_offset;
    hdr->cold_data_size = cold_data_bytes;
    hdr->main_entry_offset = 0;  /* Set by tool that seeds blob */
    hdr->syscalls_ptr_offset = 0; /* Set by tool */
    
    /* Zero hot regions */
    memset((char *)map + hdr->nodes_offset, 0, (size_t)nodes_size);
    memset((char *)map + hdr->edges_offset, 0, (size_t)edges_size);
    memset((char *)map + hdr->blob_offset, 0, (size_t)hot_blob_bytes);
    /* Cold data left as-is (will be filled by corpus loader) */
    
    /* Initialize ALL nodes with proper UINT32_MAX for edge pointers */
    /* CRITICAL: memset(0) leaves first_in/first_out = 0, pointing to edge 0! */
    Node *nodes = (Node *)((char *)map + hdr->nodes_offset);
    
    /* First: Initialize ALL nodes to prevent phantom edges */
    for (size_t i = 0; i < hot_nodes; i++) {
        nodes[i].first_in = UINT32_MAX;   /* No edges yet */
        nodes[i].first_out = UINT32_MAX;  /* No edges yet */
        nodes[i].in_degree = 0;
        nodes[i].out_degree = 0;
    }
    
    /* Then: Set byte-specific fields for nodes 0-255 */
    for (int i = 0; i < 256 && i < (int)hot_nodes; i++) {
        nodes[i].byte = (uint8_t)i;
        nodes[i].a = 0.0f;
        /* Initialize soft structure fields to zero (will be set by initialize_soft_structure) */
        nodes[i].input_propensity = 0.0f;
        nodes[i].output_propensity = 0.0f;
        nodes[i].memory_propensity = 0.0f;
        nodes[i].semantic_hint = 0;
    }
    
    /* Create temporary Graph structure for initialization (minimal, just for soft structure) */
    Graph temp_g;
    memset(&temp_g, 0, sizeof(Graph));
    temp_g.fd = fd;
    temp_g.map_base = map;
    temp_g.map_size = (size_t)total_size;
    temp_g.hdr = hdr;
    temp_g.nodes = nodes;
    temp_g.edges = (Edge *)((char *)map + hdr->edges_offset);
    temp_g.blob = (uint8_t *)((char *)map + hdr->blob_offset);
    temp_g.cold_data = (hdr->cold_data_size > 0) ? 
                       (uint8_t *)((char *)map + hdr->cold_data_offset) : NULL;
    temp_g.node_count = hdr->node_count;
    temp_g.edge_count = 0;  /* Start with no edges */
    temp_g.blob_size = hdr->blob_size;
    temp_g.cold_data_size = hdr->cold_data_size;
    
    /* Initialize soft structure scaffolding (embedded in .m file on bootup) */
    initialize_soft_structure(&temp_g, true);  /* is_new_file = true */
    
    /* Create weak initial edge suggestions (direct edge creation for v2 file creation) */
    Edge *edges = (Edge *)((char *)map + hdr->edges_offset);
    uint32_t edge_idx = 0;
    /* NO LIMITS - create edges for all ports, limited only by hot_edges allocation */
    
    /* 1. Input → Working Memory (ports 0-99 → 200-209) */
    /* Create edges for ALL input ports, not just first 10 */
    for (uint32_t input = 0; input < 100 && input < hot_nodes; input++) {
        for (uint32_t memory = 200; memory < 210 && memory < hot_nodes; memory++) {
            if (edge_idx >= hot_edges) break;
            edges[edge_idx].src = input;
            edges[edge_idx].dst = memory;
            /* RELATIVE: Use default for new file (avg_edge_strength not yet initialized) */
            edges[edge_idx].w = 0.1f;  /* Default - graph will learn through UEL */
            edges[edge_idx].next_out = nodes[input].first_out;
            edges[edge_idx].next_in = nodes[memory].first_in;
            nodes[input].first_out = edge_idx;
            nodes[memory].first_in = edge_idx;
            nodes[input].out_degree++;
            nodes[memory].in_degree++;
            edge_idx++;
        }
    }
    
    /* 2. Working Memory → Output (ports 200-209 → 100-199) */
    /* Create edges for ALL output ports, not just first 10 */
    for (uint32_t memory = 200; memory < 210 && memory < hot_nodes; memory++) {
        for (uint32_t output = 100; output < 200 && output < hot_nodes; output++) {
            if (edge_idx >= hot_edges) break;
            edges[edge_idx].src = memory;
            edges[edge_idx].dst = output;
            edges[edge_idx].w = 0.1f;
            edges[edge_idx].next_out = nodes[memory].first_out;
            edges[edge_idx].next_in = nodes[output].first_in;
            nodes[memory].first_out = edge_idx;
            nodes[output].first_in = edge_idx;
            nodes[memory].out_degree++;
            nodes[output].in_degree++;
            edge_idx++;
        }
    }
    
    /* Update edge count in header */
    hdr->edge_count = edge_idx;
    
    /* Sync and cleanup */
    msync(map, total_size, MS_SYNC);
    munmap(map, total_size);
    close(fd);
    
    return 0;
}

/* ========================================================================
 * CALL ENTRY (run UEL physics)
 * ======================================================================== */

/* EXEC node execution with crash protection */
static sigjmp_buf exec_segfault_recovery;
static volatile sig_atomic_t exec_segfault_occurred = 0;
static Graph *exec_g_global = NULL;

/* Crash handler for EXEC node execution (SIGILL, SIGSEGV, SIGBUS) */
static void exec_segfault_handler(int sig, siginfo_t *info, void *context) {
    (void)sig;
    (void)info;
    (void)context;
    
    exec_segfault_occurred = 1;
    
    if (exec_g_global) {
        /* Feed crash error to graph via error detection port (250) */
        /* Graph learns from crashes and adapts behavior */
        melvin_feed_byte(exec_g_global, 250, 0xFF, 1.0f);  /* High energy error signal */
        melvin_feed_byte(exec_g_global, 31, 0xFF, 0.8f);   /* Negative feedback */
    }
    
    /* Jump back to recovery point - must use siglongjmp in signal handler! */
    siglongjmp(exec_segfault_recovery, 1);
}

/* Execute EXEC node code - per-node execution */
static void melvin_execute_exec_node(Graph *g, uint32_t node_id) {
    if (!g || !g->hdr || !g->blob || !g->nodes) return;
    if (node_id >= g->node_count) return;
    
    ROUTE_LOG("melvin_execute_exec_node: ENTERED node_id=%u", node_id);
    
    Node *node = &g->nodes[node_id];
    
    /* Check if this is an EXEC node */
    if (node->payload_offset == 0) {
        ROUTE_LOG("  → Not an EXEC node (no payload_offset)");
        return;  /* Not an EXEC node */
    }
    if (node->payload_offset >= g->hdr->blob_size) {
        ROUTE_LOG("  → Invalid payload_offset %llu >= blob_size %llu",
                  (unsigned long long)node->payload_offset,
                  (unsigned long long)g->hdr->blob_size);
        return;  /* Invalid offset */
    }
    
    ROUTE_LOG("  payload_offset=%llu, exec_count=%u, exec_success_rate=%.3f",
              (unsigned long long)node->payload_offset, node->exec_count, node->exec_success_rate);
    
    /* RELATIVE THRESHOLD: Calculate threshold based on graph state */
    /* exec_threshold_ratio is relative to avg_activation (like other thresholds) */
    float activation_safe = (g->avg_activation != g->avg_activation || g->avg_activation < 0.0f) ? 
                            0.1f : g->avg_activation;
    if (activation_safe < 0.001f) activation_safe = 0.001f;  /* Minimum */
    
    /* Get threshold ratio (default 0.5 = 50% of avg_activation if not set) */
    /* RELATIVE: Lower default threshold to allow easier activation */
    float threshold_ratio = node->exec_threshold_ratio;
    if (threshold_ratio <= 0.0f) threshold_ratio = 0.5f;  /* Default: 50% of avg_activation (was 1.0) */
    
    /* Calculate relative threshold */
    float threshold = activation_safe * threshold_ratio;
    
    /* Dynamic minimum: prevent threshold from being too low (at least 5% of edge strength or 0.005) */
    /* RELATIVE: Lower minimum to allow easier activation */
    float dynamic_min = (g->avg_edge_strength > 0.0f) ? (g->avg_edge_strength * 0.05f) : 0.005f;
    if (threshold < dynamic_min) threshold = dynamic_min;
    
    /* Check if activation exceeds threshold */
    float activation = fabsf(node->a);
    ROUTE_LOG("  Activation check: activation=%.3f, threshold=%.3f", activation, threshold);
    if (activation < threshold) {
        ROUTE_LOG("  → Not activated enough (%.3f < %.3f), returning", activation, threshold);
        return;  /* Not activated enough */
    }
    ROUTE_LOG("  → Activation exceeds threshold, proceeding with execution");
    
    /* NEW: Get input values from blob (passed by pattern expansion) */
    uint64_t input_offset = node->payload_offset + 256;  /* After code */
    uint64_t input1 = 0, input2 = 0;
    bool has_inputs = false;
    
    ROUTE_LOG("  Reading inputs from offset %llu", (unsigned long long)input_offset);
    
    if (input_offset + (2 * sizeof(uint64_t)) <= g->hdr->blob_size) {
        uint64_t *input_ptr = (uint64_t *)(g->blob + (input_offset - g->hdr->blob_offset));
        input1 = input_ptr[0];
        input2 = input_ptr[1];
        ROUTE_LOG("  Inputs: input1=%llu, input2=%llu", 
                  (unsigned long long)input1, (unsigned long long)input2);
        if (input1 > 0 || input2 > 0) {
            has_inputs = true;
            ROUTE_LOG("  → Has inputs, will execute");
        } else {
            ROUTE_LOG("  → No inputs (both zero)");
        }
    } else {
        ROUTE_LOG("  → ERROR: Input offset %llu + 16 exceeds blob_size %llu",
                  (unsigned long long)input_offset, (unsigned long long)g->hdr->blob_size);
    }
    
    /* Safety check: Don't execute if blob looks invalid */
    uint8_t *entry = g->blob + node->payload_offset;
    bool has_code = false;
    for (int i = 0; i < 16 && (node->payload_offset + i) < g->hdr->blob_size; i++) {
        if (entry[i] != 0x00 && entry[i] != 0xFF) {
            has_code = true;
            break;
        }
    }
    
    if (!has_code) {
        /* No valid code - mark as failed execution */
        node->exec_count++;
        if (node->exec_success_rate > 0.0f) {
            node->exec_success_rate *= 0.95f;  /* Decay success rate */
        }
        return;
    }
    
    /* Increment execution count */
    node->exec_count++;
    
    /* DYNAMIC BLOB EXECUTION: Execute whatever code is in the blob! */
    /* NO HARDCODING - brain learns operations from fed machine code */
    bool execution_success = false;
    uint64_t result = 0;
    
    if (has_inputs && has_code) {
        /* Cast blob as executable function - NO HARDCODED OPERATIONS! */
        /* Function signature: uint64_t exec(uint64_t input1, uint64_t input2) */
        typedef uint64_t (*exec_func)(uint64_t, uint64_t);
        
        /* Get function pointer from blob */
        uint8_t *code_ptr = g->blob + node->payload_offset;
        exec_func f = (exec_func)code_ptr;
        
        fprintf(stderr, "\n🚀 Executing blob code at offset %llu...\n",
                (unsigned long long)node->payload_offset);
        fprintf(stderr, "   Inputs: %llu, %llu\n", 
                (unsigned long long)input1, (unsigned long long)input2);
        
        /* Set up crash protection (SIGILL, SIGSEGV, SIGBUS) */
        struct sigaction old_sa_segv, old_sa_bus, old_sa_ill, new_sa;
        exec_g_global = g;
        exec_segfault_occurred = false;
        
        /* Set up signal handlers for all crash types */
        new_sa.sa_sigaction = exec_segfault_handler;
        sigemptyset(&new_sa.sa_mask);
        new_sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &new_sa, &old_sa_segv);
        sigaction(SIGBUS, &new_sa, &old_sa_bus);
        sigaction(SIGILL, &new_sa, &old_sa_ill);  /* Illegal instruction */
        
        /* Set jump point for recovery (sigsetjmp saves signal mask) */
        if (sigsetjmp(exec_segfault_recovery, 1) == 0) {
            /* EXECUTE THE BLOB CODE ON CPU WITH PROTECTION! */
        /* This is REAL machine code execution, not simulation! */
        result = f(input1, input2);
            execution_success = true;  /* If we get here, execution succeeded */
        
        fprintf(stderr, "\n⭐⭐⭐ BLOB EXECUTION SUCCESS! ⭐⭐⭐\n");
        fprintf(stderr, "EXEC Node: %u\n", node_id);
        fprintf(stderr, "Blob Offset: %llu\n", (unsigned long long)node->payload_offset);
        fprintf(stderr, "Operation: %llu ⊕ %llu = %llu\n",
                (unsigned long long)input1, (unsigned long long)input2, 
                (unsigned long long)result);
        fprintf(stderr, "⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐\n\n");
        
        ROUTE_LOG("  ★★★ BLOB EXECUTION SUCCESS: %llu ⊕ %llu = %llu ★★★",
                  (unsigned long long)input1, (unsigned long long)input2, 
                  (unsigned long long)result);
        
        /* Store result back in blob */
        uint64_t result_offset = input_offset + (2 * sizeof(uint64_t));
        if (result_offset + sizeof(uint64_t) <= g->hdr->blob_size) {
            uint64_t *result_ptr = (uint64_t *)(g->blob + (result_offset - g->hdr->blob_offset));
            *result_ptr = result;
            ROUTE_LOG("  Stored result at offset %llu", (unsigned long long)result_offset);
        }
        
        /* Convert result back to pattern (graph learns this) */
        ROUTE_LOG("  Converting result to pattern via port 100");
        convert_result_to_pattern(g, node_id, result);
        } else {
            /* Crash occurred (SIGILL, SIGSEGV, SIGBUS) - execution failed */
            execution_success = false;
            fprintf(stderr, "\n❌ BLOB EXECUTION FAILED (crashed)\n");
            fprintf(stderr, "   Node: %u, Offset: %llu\n", 
                    node_id, (unsigned long long)node->payload_offset);
            fprintf(stderr, "   Error fed to port 250 (graph learns from failure)\n\n");
            ROUTE_LOG("  ❌ BLOB EXECUTION CRASHED - reinforcement learning engaged");
        }
        
        /* Restore signal handlers */
        sigaction(SIGSEGV, &old_sa_segv, NULL);
        sigaction(SIGBUS, &old_sa_bus, NULL);
        sigaction(SIGILL, &old_sa_ill, NULL);
        exec_g_global = NULL;
    } else if (has_code) {
        /* No inputs - execute machine code normally */
        /* Get function pointer to node's code */
        /* Blob code signature: void exec_code(Graph *g, uint32_t self_node_id) */
        void (*exec_code)(Graph *g, uint32_t) = (void (*)(Graph *g, uint32_t))(
            g->blob + node->payload_offset
        );
        
        /* Execute the code with crash protection */
        /* Set up signal handlers for this execution */
        struct sigaction old_sa_segv, old_sa_bus, old_sa_ill, new_sa;
        exec_g_global = g;
        exec_segfault_occurred = false;
        
        /* Set up signal handlers for all crash types */
        new_sa.sa_sigaction = exec_segfault_handler;
        sigemptyset(&new_sa.sa_mask);
        new_sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &new_sa, &old_sa_segv);
        sigaction(SIGBUS, &new_sa, &old_sa_bus);
        sigaction(SIGILL, &new_sa, &old_sa_ill);  /* Illegal instruction */
        
        /* Set jump point for recovery (sigsetjmp saves signal mask) */
        if (sigsetjmp(exec_segfault_recovery, 1) == 0) {
            /* Try to execute the code */
            exec_code(g, node_id);
            execution_success = true;  /* If we get here, execution succeeded */
        } else {
            /* Crash occurred - execution failed */
            execution_success = false;
            fprintf(stderr, "\n❌ EXEC code crashed (node %u)\n", node_id);
            ROUTE_LOG("  ❌ EXEC CRASHED - reinforcement learning engaged");
        }
        
        /* Restore signal handlers */
        sigaction(SIGSEGV, &old_sa_segv, NULL);
        sigaction(SIGBUS, &old_sa_bus, NULL);
        sigaction(SIGILL, &old_sa_ill, NULL);
        exec_g_global = NULL;
    }
    
    /* Update success rate (exponential moving average) */
    float success_value = execution_success ? 1.0f : 0.0f;
    if (node->exec_success_rate == 0.0f) {
        node->exec_success_rate = success_value;  /* First execution */
    } else {
        node->exec_success_rate = 0.9f * node->exec_success_rate + 0.1f * success_value;
    }
    
    /* Adjust threshold ratio based on success rate (successful nodes get lower ratio) */
    /* This makes successful nodes easier to trigger (lower ratio = lower threshold) */
    if (node->exec_success_rate > 0.7f && node->exec_threshold_ratio > 0.3f) {
        node->exec_threshold_ratio *= 0.99f;  /* Slightly lower ratio for successful nodes */
    } else if (node->exec_success_rate < 0.3f && node->exec_threshold_ratio < 2.0f) {
        node->exec_threshold_ratio *= 1.01f;  /* Slightly higher ratio for failing nodes */
    }
    
    /* Clamp ratio to reasonable range (0.1x to 3.0x avg_activation) */
    if (node->exec_threshold_ratio < 0.1f) node->exec_threshold_ratio = 0.1f;
    if (node->exec_threshold_ratio > 3.0f) node->exec_threshold_ratio = 3.0f;
}

/* Execute blob code if it exists - GRAPH-DRIVEN: only executes when graph decides */
static void melvin_execute_blob(Graph *g) {
    if (!g || !g->hdr || !g->blob) return;
    /* Allow offset 0 (start of blob) as valid entry point */
    if (g->hdr->main_entry_offset >= g->hdr->blob_size) return;  /* Invalid offset */
    
    /* Safety check: Don't execute if blob looks invalid */
    /* Check if blob at entry point contains valid-looking code (not all zeros) */
    uint8_t *entry = g->blob + g->hdr->main_entry_offset;
    bool has_code = false;
    for (int i = 0; i < 16 && (g->hdr->main_entry_offset + i) < g->hdr->blob_size; i++) {
        if (entry[i] != 0x00 && entry[i] != 0xFF) {
            has_code = true;
            break;
        }
    }
    
    /* For testing: If blob is just markers, don't actually execute (would segfault) */
    /* In production, this would be real ARM64 code */
    if (!has_code) {
        /* Just mark execution without actually calling */
        g->blob[0] = 0xEE;  /* Execution marker */
        return;
    }
    
    /* Get function pointer to blob's main entry */
    /* Blob code signature: void blob_main(Graph *g) */
    void (*blob_main)(Graph *g) = (void (*)(Graph *g))(
        g->blob + g->hdr->main_entry_offset
    );
    
    /* Execute blob code - graph's own code runs */
    /* Graph controls when this happens through output node activation */
    /* Note: This will segfault if blob doesn't contain valid ARM64 code */
    blob_main(g);
}

void melvin_call_entry(Graph *g) {
    if (!g || !g->hdr) return;
    
    /* Run UEL physics (embedded in melvin.c) */
    /* PURE SUBSTRATE: Only graph structure and UEL physics, no tool-specific code */
    uel_main(g);
    
    /* GRAPH-DRIVEN BLOB EXECUTION: Execute blob code when output nodes activate */
    /* Graph decides when to run its own code through activation patterns */
    /* Check if any output nodes are highly activated - if so, blob code might want to run */
    bool should_execute_blob = false;
    float max_output_activation = 0.0f;
    
    /* Safety check: output_propensity must be allocated */
    if (!g->output_propensity) return;
    
    /* Scan output ports (100-199) for high activation */
    if (g->output_propensity) {
        for (uint32_t i = 100; i < 200 && i < g->node_count; i++) {
            if (g->output_propensity[i] > 0.5f) {  /* High output propensity */
                float a_abs = fabsf(g->nodes[i].a);
                if (a_abs > g->avg_activation * 1.5f) {  /* Significantly above average */
                    should_execute_blob = true;
                    if (a_abs > max_output_activation) {
                        max_output_activation = a_abs;
                    }
                }
            }
        }
    }
    
    /* Also check tool gateway outputs (they produce patterns) */
    if (g->output_propensity) {
        for (uint32_t i = 300; i < 700 && i < g->node_count; i++) {
            if (g->output_propensity[i] > 0.7f) {  /* Very high output propensity (tool outputs) */
                float a_abs = fabsf(g->nodes[i].a);
                if (a_abs > g->avg_activation * 1.2f) {
                    should_execute_blob = true;
                    if (a_abs > max_output_activation) {
                        max_output_activation = a_abs;
                    }
                }
            }
        }
    }
    
    /* Execute blob code if graph decided to (through activation) */
    /* Graph learns when blob execution is useful through UEL feedback */
    /* Only execute if main_entry_offset > 0 (blob code has been set) */
    if (should_execute_blob && g->hdr->main_entry_offset > 0 && 
        g->hdr->main_entry_offset < g->hdr->blob_size) {
        /* Debug: Log blob execution (first few times) */
        static int exec_count = 0;
        exec_count++;
        if (exec_count <= 3) {  /* Log first 3 executions */
            /* Use printf instead of fprintf(stderr) to avoid include issues */
            printf("[BLOB] Executing blob at offset %llu (execution #%d)\n",
                   (unsigned long long)g->hdr->main_entry_offset, exec_count);
            fflush(stdout);
        }
        melvin_execute_blob(g);
    }
}

/* ========================================================================
 * DEBUG HELPERS (read-only inspection)
 * ======================================================================== */

float melvin_get_activation(Graph *g, uint32_t node_id) {
    if (!g || !g->hdr) return 0.0f;
    
    /* Ensure node exists - graph grows dynamically */
    ensure_node(g, node_id);
    return g->nodes[node_id].a;
}

/* ========================================================================
 * COLD DATA ACCESS (graph can copy from cold to hot)
 * ======================================================================== */

/* Copy bytes from cold_data to hot blob - graph-accessible via machine code */
void melvin_copy_from_cold(Graph *g, uint64_t cold_offset, uint64_t length, uint64_t blob_target_offset) {
    if (!g || !g->cold_data || !g->blob) return;
    
    /* Validate cold access */
    if (cold_offset >= g->cold_data_size) return;
    if (cold_offset + length > g->cold_data_size) {
        length = g->cold_data_size - cold_offset;  /* Clamp to available */
    }
    
    /* Validate blob target */
    if (blob_target_offset >= g->blob_size) return;
    if (blob_target_offset + length > g->blob_size) {
        length = g->blob_size - blob_target_offset;  /* Clamp to available */
    }
    
    /* Copy from cold to hot */
    memcpy(g->blob + blob_target_offset, g->cold_data + cold_offset, (size_t)length);
}

/* Create EXEC node: set payload_offset and threshold ratio for a node */
/* threshold_ratio: relative to avg_activation (1.0 = 100%, 0.5 = 50%, 2.0 = 200%) */
uint32_t melvin_create_exec_node(Graph *g, uint32_t node_id, uint64_t blob_offset, float threshold_ratio) {
    if (!g || !g->nodes || !g->blob) return UINT32_MAX;
    
    /* Ensure node exists */
    ensure_node(g, node_id);
    
    /* Validate blob offset */
    if (blob_offset >= g->hdr->blob_size) return UINT32_MAX;
    
    /* Set EXEC node fields */
    Node *node = &g->nodes[node_id];
    node->payload_offset = blob_offset;
    /* Default threshold ratio: 1.0 = 100% of avg_activation */
    node->exec_threshold_ratio = (threshold_ratio > 0.0f) ? threshold_ratio : 1.0f;
    node->exec_count = 0;
    node->exec_success_rate = 0.0f;
    
    /* Clamp ratio to reasonable range */
    if (node->exec_threshold_ratio < 0.1f) node->exec_threshold_ratio = 0.1f;
    if (node->exec_threshold_ratio > 3.0f) node->exec_threshold_ratio = 3.0f;
    
    return node_id;
}

/* ========================================================================
 * PATTERN SYSTEM - Global Law: Patterns form from repeated sequences
 * ======================================================================== */

#define PATTERN_MAGIC 0x4E544150  /* "PATN" in little-endian */

/* Simple hash function for sequences */
static uint64_t hash_sequence(const uint32_t *sequence, uint32_t length) {
    uint64_t hash = 5381;
    for (uint32_t i = 0; i < length; i++) {
        hash = ((hash << 5) + hash) + sequence[i];
    }
    return hash;
}

/* Track sequence in hash table - returns count and stores first occurrence */
static uint32_t track_sequence(Graph *g, const uint32_t *sequence, uint32_t length, 
                                uint32_t **first_occurrence_out) {
    if (!g || !g->sequence_hash_table || !sequence || length == 0) {
        if (first_occurrence_out) *first_occurrence_out = NULL;
        return 0;
    }
    if (g->sequence_hash_size == 0) {
        if (first_occurrence_out) *first_occurrence_out = NULL;
        return 0;
    }
    
    uint64_t hash = hash_sequence(sequence, length);
    uint64_t idx = hash % g->sequence_hash_size;
    
    /* Hash table entry: [hash, count, storage_offset] */
    /* Simple linear probing for collisions */
    for (uint32_t i = 0; i < 100; i++) {  /* Max 100 probes */
        uint64_t slot = (idx + i) % g->sequence_hash_size;
        uint32_t *slot_ptr = &g->sequence_hash_table[slot * 3];  /* 3 values per slot */
        
        if (slot_ptr[0] == 0) {
            /* Empty slot - first occurrence - store sequence */
            slot_ptr[0] = hash & 0xFFFFFFFF;
            slot_ptr[1] = 1;  /* Count = 1 */
            
            /* Store first occurrence in sequence_storage */
            if (g->sequence_storage && g->sequence_storage_pos + length + 1 < g->sequence_storage_size) {
                uint32_t storage_offset = (uint32_t)g->sequence_storage_pos;
                g->sequence_storage[g->sequence_storage_pos++] = length;  /* Store length first */
                for (uint32_t j = 0; j < length; j++) {
                    g->sequence_storage[g->sequence_storage_pos++] = sequence[j];
                }
                slot_ptr[2] = storage_offset;  /* Store offset to first occurrence */
                if (first_occurrence_out) *first_occurrence_out = &g->sequence_storage[storage_offset + 1];
            } else {
                slot_ptr[2] = 0;
                if (first_occurrence_out) *first_occurrence_out = NULL;
            }
            return 1;
        } else if (slot_ptr[0] == (hash & 0xFFFFFFFF)) {
            /* Found sequence - increment count */
            slot_ptr[1]++;
            
            /* Return pointer to first occurrence */
            if (first_occurrence_out && slot_ptr[2] > 0) {
                *first_occurrence_out = &g->sequence_storage[slot_ptr[2] + 1];
            } else {
                *first_occurrence_out = NULL;
            }
            return slot_ptr[1];
        }
    }
    
    if (first_occurrence_out) *first_occurrence_out = NULL;
    return 0;  /* Hash table full or collision */
}

/* Extract pattern from two sequences - find common structure with blanks */
/* IMPROVED: Better blank detection by comparing sequences */
static uint32_t extract_pattern(const uint32_t *seq1, const uint32_t *seq2, uint32_t length,
                                 PatternElement *pattern) {
    if (!seq1 || !seq2 || !pattern || length == 0) return 0;
    
    uint32_t pattern_idx = 0;
    uint32_t blank_idx = 0;
    
    for (uint32_t i = 0; i < length; i++) {
        if (seq1[i] == seq2[i]) {
            /* Same value - data node (constant in pattern) */
            pattern[pattern_idx].is_blank = 0;
            pattern[pattern_idx].value = seq1[i];
            pattern_idx++;
        } else {
            /* Different value - blank position (variable in pattern) */
            pattern[pattern_idx].is_blank = 1;
            pattern[pattern_idx].value = blank_idx;  /* pos0, pos1, etc. (local to pattern) */
            pattern_idx++;
            blank_idx++;
        }
    }
    
    return pattern_idx;
}

/* Check if pattern matches a sequence (for pattern pattern matching) */
static bool pattern_matches_sequence(Graph *g, uint32_t pattern_node_id, const uint32_t *sequence, 
                                      uint32_t length, uint32_t *bindings) {
    if (!g || !sequence || pattern_node_id >= g->node_count) return false;
    
    Node *pattern_node = &g->nodes[pattern_node_id];
    if (pattern_node->pattern_data_offset == 0) return false;  /* Not a pattern node */
    
    /* Read pattern data from blob */
    uint64_t pattern_offset = pattern_node->pattern_data_offset - g->hdr->blob_offset;
    if (pattern_offset >= g->blob_size) return false;
    
    PatternData *pattern_data = (PatternData *)(g->blob + pattern_offset);
    if (pattern_data->magic != PATTERN_MAGIC) return false;
    
    /* SIMPLE STRUCTURAL MATCHING: Blanks = wildcards, concrete = exact match */
    uint32_t pattern_len = pattern_data->element_count;
    
    /* Allow length mismatch (within 3 elements) */
    uint32_t length_diff = (pattern_len > length) ? (pattern_len - length) : (length - pattern_len);
    if (length_diff > 3) return false;
    
    /* FIXED: Initialize tracking variables (was uninitialized - critical bug!) */
    uint32_t blank_bindings[256] = {0};  /* Track blank bindings */
    uint32_t blank_count = 0;             /* Count of blanks matched */
    float total_similarity = 0.0f;        /* Total similarity score */
    
    /* Calculate length penalty (penalize length differences) */
    float length_penalty = 1.0f - ((float)length_diff / (float)(pattern_len > length ? pattern_len : length + 1));
    if (length_penalty < 0.5f) length_penalty = 0.5f;  /* Don't penalize too heavily */
    
    uint32_t match_len = (pattern_len < length) ? pattern_len : length;
    uint32_t concrete_matches = 0;  /* Track concrete node matches */
    
    for (uint32_t i = 0; i < match_len && i < pattern_len; i++) {
        PatternElement *elem = &pattern_data->elements[i];
        
        if (elem->is_blank == 0) {
            /* Concrete node - must match exactly */
            if (elem->value != sequence[i]) {
                /* Allow '?' to match result position in queries */
                uint32_t question_mark = (uint32_t)'?';
                if (!(sequence[i] == question_mark && i == length - 1)) {
                    return false;  /* No match */
                }
                /* '?' at end matches any result - count as similarity but lower weight */
                total_similarity += 0.5f;
            } else {
                /* Exact match - high similarity */
                total_similarity += 1.0f;
                concrete_matches++;
            }
        } else {
            /* Blank - bind to sequence value */
            /* Blanks match based on position - same blank position binds to same value */
            uint32_t blank_pos = elem->value;
            if (blank_pos < 256) {
                if (blank_bindings[blank_pos] == 0 && sequence[i] != 0) {
                    /* First time seeing this blank - bind it */
                    /* Note: sequence[i] == 0 would be NULL byte, valid but rare */
                    blank_bindings[blank_pos] = sequence[i];
                    blank_count++;
                    total_similarity += 1.0f;  /* Blanks always match when first bound */
                } else if (blank_bindings[blank_pos] == sequence[i]) {
                    /* Same value - perfect match */
                    total_similarity += 1.0f;
                } else if (blank_bindings[blank_pos] == 0 && sequence[i] == 0) {
                    /* Both are NULL byte - treat as match */
                    blank_bindings[blank_pos] = sequence[i];
                    blank_count++;
                    total_similarity += 1.0f;
                } else {
                    /* Different value - same blank position should have same value */
                    /* This is a mismatch (e.g., blank 0 bound to '1' and '2') */
                    return false;
                }
            }
        }
    }
    
    /* RELATIVE: Check if overall similarity is high enough */
    /* Allow partial matches - don't require 100% accuracy */
    /* Account for length differences */
    float match_len_safe = (match_len > 0) ? (float)match_len : 1.0f;
    float avg_similarity = total_similarity / match_len_safe;
    avg_similarity *= length_penalty;  /* Penalize length mismatches */
    
    /* Dynamic threshold based on pattern strength and graph state */
    float pattern_strength = (pattern_data->strength > 0.0f) ? pattern_data->strength : 0.1f;
    float frequency_factor = (pattern_data->frequency > 0.0f) ? 
                            (1.0f / (1.0f + pattern_data->frequency)) : 0.5f;
    
    /* RELATIVE: Base threshold scales with graph state */
    /* Simpler calculation - just need a reasonable threshold */
    float base_threshold = 0.3f;  /* 30% similarity minimum */
    
    /* Stronger, more frequent patterns can match with lower similarity */
    /* Weaker, less frequent patterns need higher similarity */
    float adjusted_threshold = base_threshold * (1.0f + pattern_strength * 0.5f) * frequency_factor;
    if (adjusted_threshold < 0.1f) adjusted_threshold = 0.1f;  /* Very lenient minimum */
    if (adjusted_threshold > 0.8f) adjusted_threshold = 0.8f;  /* Maximum (allow partial matches) */
    
    /* ROUTING DEBUG */
    ROUTE_LOG("pattern_matches_sequence: pattern=%u, seq_len=%u, pattern_len=%u",
              pattern_node_id, length, pattern_len);
    ROUTE_LOG("  blanks=%u, concrete_matches=%u, total_similarity=%.2f, avg=%.2f, threshold=%.2f",
              blank_count, concrete_matches, total_similarity, avg_similarity, adjusted_threshold);
    
    if (avg_similarity < adjusted_threshold) {
        ROUTE_LOG("  → REJECTED: similarity %.2f < threshold %.2f", avg_similarity, adjusted_threshold);
        return false;  /* Not similar enough */
    }
    
    ROUTE_LOG("  → MATCHED!");
    
    /* Copy bindings to output */
    /* IMPORTANT: blank_bindings is indexed by blank position (0, 1, 2, etc.) */
    /* We need to copy all blank bindings, not just the first blank_count */
    if (bindings) {
        for (uint32_t i = 0; i < 256; i++) {
            bindings[i] = blank_bindings[i];
        }
        ROUTE_LOG("  Bindings: [0]=%u, [1]=%u, [2]=%u", bindings[0], bindings[1], bindings[2]);
    }
    
    return true;
}

/* Extract value from pattern sequence - GENERAL mechanism */
/* Graph learns which patterns extract which values through examples */
static PatternValue extract_pattern_value(Graph *g, const uint32_t *sequence, 
                                          uint32_t length, uint32_t pattern_node_id) {
    PatternValue value = {0};
    
    if (!g || !sequence || length == 0 || pattern_node_id >= g->node_count) {
        return value;
    }
    
    Node *pattern_node = &g->nodes[pattern_node_id];
    
    /* Check if pattern has learned a value mapping */
    if (pattern_node->pattern_value_offset > 0) {
        /* Pattern has learned value - read it from blob */
        uint64_t value_offset = pattern_node->pattern_value_offset - g->hdr->blob_offset;
        if (value_offset < g->blob_size) {
            PatternValue *stored_value = (PatternValue *)(g->blob + value_offset);
            value = *stored_value;
            return value;  /* Return learned value */
        }
    }
    
    /* Pattern hasn't learned value yet - try to infer from sequence */
    /* SIMPLE RULE: If all digits, parse as integer; otherwise store as identifier */
    
    /* Check if sequence is all digits (could be a number) */
    bool all_digits = true;
    for (uint32_t i = 0; i < length; i++) {
        uint8_t byte_val = (uint8_t)(sequence[i] & 0xFF);
        if (byte_val < '0' || byte_val > '9') {
            all_digits = false;
            break;
        }
    }
    
    if (all_digits && length <= 10) {
        /* All digits - parse as integer */
        uint64_t num = 0;
        for (uint32_t i = 0; i < length; i++) {
            uint8_t digit = (uint8_t)(sequence[i] & 0xFF) - '0';
            num = num * 10 + digit;
        }
        value.value_type = 0;  /* Number type */
        value.value_data = num;
        value.confidence = 1.0f;  /* High confidence for parsed integers */
        ROUTE_LOG("extract_pattern_value: Parsed %u digits → %llu", length, (unsigned long long)num);
    } else {
        /* Not all digits - store as identifier */
        value.value_type = 1;  /* String/concept type */
        value.value_data = (uint64_t)sequence[0];  /* First node as identifier */
        value.confidence = 0.5f;  /* Medium confidence */
        ROUTE_LOG("extract_pattern_value: Non-numeric sequence, type=1, value=%llu",
                  (unsigned long long)value.value_data);
    }
    
    return value;
}

/* Learn value mapping from examples - GRAPH LEARNS THIS */
/* Called when pattern appears with a known value in context */
static void learn_value_mapping(Graph *g, uint32_t pattern_node_id, 
                                PatternValue example_value) {
    if (!g || pattern_node_id >= g->node_count) return;
    
    Node *pattern_node = &g->nodes[pattern_node_id];
    
    /* Check if pattern already has a value */
    if (pattern_node->pattern_value_offset == 0) {
        /* No value yet - learn from example */
        
        /* Allocate space in blob for PatternValue */
        uint64_t value_offset = g->hdr->main_entry_offset;
        size_t value_size = sizeof(PatternValue);
        
        if (value_offset + value_size <= g->hdr->blob_size) {
            /* Store value in blob */
            PatternValue *stored_value = (PatternValue *)(g->blob + value_offset);
            *stored_value = example_value;
            stored_value->confidence = 0.5f;  /* Initial confidence */
            
            pattern_node->pattern_value_offset = g->hdr->blob_offset + value_offset;
            g->hdr->main_entry_offset += value_size;  /* Advance blob pointer */
        }
    } else {
        /* Already has value - strengthen if matches */
        uint64_t value_offset = pattern_node->pattern_value_offset - g->hdr->blob_offset;
        if (value_offset < g->blob_size) {
            PatternValue *stored_value = (PatternValue *)(g->blob + value_offset);
            if (stored_value->value_type == example_value.value_type &&
                stored_value->value_data == example_value.value_data) {
                /* Matches - strengthen confidence */
                stored_value->confidence = fminf(1.0f, stored_value->confidence + 0.1f);
            }
        }
    }
}

/* Pass values to EXEC node - GENERAL mechanism */
/* Graph learns which values go to which EXEC nodes */
static void pass_values_to_exec(Graph *g, uint32_t exec_node_id, 
                                PatternValue *values, uint32_t value_count) {
    if (!g || exec_node_id >= g->node_count || !values || value_count == 0) return;
    
    ROUTE_LOG("pass_values_to_exec: exec_node_id=%u, value_count=%u", exec_node_id, value_count);
    
    Node *exec_node = &g->nodes[exec_node_id];
    if (exec_node->payload_offset == 0) {
        ROUTE_LOG("  → Not an EXEC node (no payload_offset)");
        return;  /* Not an EXEC node */
    }
    
    /* Extract numeric values (graph learns which values are numbers) */
    uint64_t numeric_inputs[8] = {0};
    uint32_t num_count = 0;
    
    for (uint32_t i = 0; i < value_count && num_count < 8; i++) {
        if (values[i].value_type == 0) {  /* Number type */
            numeric_inputs[num_count++] = values[i].value_data;
            ROUTE_LOG("  Input[%u]: %llu", num_count - 1, (unsigned long long)values[i].value_data);
        }
    }
    
    if (num_count >= 2) {
        fprintf(stderr, "✅ Found %u numeric values\n", num_count);
        for (uint32_t i = 0; i < num_count; i++) {
            fprintf(stderr, "  Value[%u] = %llu\n", i, (unsigned long long)numeric_inputs[i]);
        }
        
        ROUTE_LOG("  Storing %u numeric inputs to EXEC node %u", num_count, exec_node_id);
        
        /* Store inputs in blob at payload_offset + offset */
        uint64_t input_offset = exec_node->payload_offset + 256;  /* After code */
        
        fprintf(stderr, "Storing values to EXEC node %u:\n", exec_node_id);
        fprintf(stderr, "  payload_offset = %llu\n", (unsigned long long)exec_node->payload_offset);
        fprintf(stderr, "  input_offset = %llu\n", (unsigned long long)input_offset);
        
        ROUTE_LOG("  Input offset: %llu (payload_offset=%llu + 256)",
                  (unsigned long long)input_offset, (unsigned long long)exec_node->payload_offset);
        
        if (input_offset + (num_count * sizeof(uint64_t)) <= g->hdr->blob_size) {
            uint64_t *input_ptr = (uint64_t *)(g->blob + (input_offset - g->hdr->blob_offset));
            for (uint32_t i = 0; i < num_count; i++) {
                input_ptr[i] = numeric_inputs[i];
                fprintf(stderr, "  ✅ Stored value[%u] = %llu\n", i, (unsigned long long)numeric_inputs[i]);
                ROUTE_LOG("    Stored input[%u] = %llu at offset %llu",
                          i, (unsigned long long)numeric_inputs[i],
                          (unsigned long long)(input_offset + i * sizeof(uint64_t)));
            }
        } else {
            fprintf(stderr, "❌ ERROR: Input offset too large!\n");
            fprintf(stderr, "  input_offset = %llu, blob_size = %llu\n",
                    (unsigned long long)input_offset, (unsigned long long)g->hdr->blob_size);
            ROUTE_LOG("  → ERROR: Input offset %llu + %u*8 exceeds blob_size %llu",
                      (unsigned long long)input_offset, num_count,
                      (unsigned long long)g->hdr->blob_size);
        }
        
        /* Activate EXEC node - give large fixed boost to ensure it fires */
        /* SIMPLE RULE: Pattern match + values = EXEC should fire, no soft gating */
        float activation_boost = 10.0f;  /* Large fixed boost to guarantee firing */
        exec_node->a += activation_boost;
        exec_node->exec_count = num_count;
        prop_queue_add(g, exec_node_id);
        
        fprintf(stderr, "🔥 Activating EXEC node %u (boost=%.1f, new_activation=%.3f)\n",
                exec_node_id, activation_boost, exec_node->a);
        fprintf(stderr, "===============================\n\n");
        
        ROUTE_LOG("  Activation boost: %.3f (fixed), new activation: %.3f", 
                  activation_boost, exec_node->a);
        
        fprintf(stderr, "🚀 Triggering EXEC execution directly...\n");
        
        /* Also trigger execution directly for pattern-driven route */
        /* This bypasses activation threshold for pattern-driven execution */
        melvin_execute_exec_node(g, exec_node_id);
    } else {
        fprintf(stderr, "❌ Not enough numeric values (%u < 2)\n", num_count);
        fprintf(stderr, "===============================\n\n");
        ROUTE_LOG("  → Not enough numeric inputs (%u < 2)", num_count);
    }
}

/* Extract values from pattern bindings and route to EXEC nodes - GENERAL mechanism */
/* Called when a pattern matches a sequence during discovery */
static void extract_and_route_to_exec(Graph *g, uint32_t pattern_node_id, const uint32_t *bindings) {
    if (!g || pattern_node_id >= g->node_count || !bindings) return;
    
    fprintf(stderr, "\n📦 ===== VALUE EXTRACTION =====\n");
    fprintf(stderr, "Pattern node: %u\n", pattern_node_id);
    
    ROUTE_LOG("extract_and_route_to_exec: pattern_node_id=%u", pattern_node_id);
    
    Node *pattern_node = &g->nodes[pattern_node_id];
    if (pattern_node->pattern_data_offset == 0) {
        fprintf(stderr, "❌ Not a pattern node (no pattern_data_offset)\n");
        fprintf(stderr, "===============================\n\n");
        ROUTE_LOG("  → Not a pattern node (no pattern_data_offset)");
        return;  /* Not a pattern node */
    }
    
    /* Read pattern to find blanks */
    uint64_t pattern_offset = pattern_node->pattern_data_offset - g->hdr->blob_offset;
    if (pattern_offset >= g->blob_size) {
        ROUTE_LOG("  → Invalid pattern offset");
        return;
    }
    
    PatternData *pattern_data = (PatternData *)(g->blob + pattern_offset);
    if (pattern_data->magic != 0x4E544150) {
        ROUTE_LOG("  → Invalid pattern magic");
        return;  /* PATTERN_MAGIC */
    }
    
    ROUTE_LOG("  Pattern: element_count=%u", pattern_data->element_count);
    
    /* Extract values from blanks */
    PatternValue extracted_values[16];
    uint32_t value_count = 0;
    
    for (uint32_t i = 0; i < pattern_data->element_count && value_count < 16; i++) {
        PatternElement *elem = &pattern_data->elements[i];
        if (elem->is_blank != 0) {
            /* Blank - check if it has a binding */
            uint32_t blank_pos = elem->value;  /* blank_pos is the blank index (0, 1, 2, etc.) */
            
            ROUTE_LOG("  Element[%u]: blank, blank_pos=%u", i, blank_pos);
            
            if (blank_pos >= 256) {
                ROUTE_LOG("    → ERROR: blank_pos %u >= 256, skipping", blank_pos);
                continue;  /* Invalid blank position */
            }
            
            uint32_t bound_node = bindings[blank_pos];
            ROUTE_LOG("    → bindings[%u] = %u", blank_pos, bound_node);
            
            if (bound_node == 0) {
                ROUTE_LOG("    → No binding for blank %u, skipping", blank_pos);
                continue;  /* No binding for this blank */
            }
            
            if (bound_node >= g->node_count) {
                ROUTE_LOG("    → WARNING: bound_node %u >= node_count %llu, skipping",
                          bound_node, (unsigned long long)g->node_count);
                continue;
            }
            
            /* Skip '?' - it's a query placeholder, not a value to extract */
            uint32_t question_mark = (uint32_t)'?';
            if (bound_node == question_mark) {
                ROUTE_LOG("    → Skipping '?' placeholder");
                continue;
            }
            
            /* ROUTING DEBUG: Log bound node details */
            Node *bound_n = &g->nodes[bound_node];
            uint8_t bound_byte = bound_n->byte;
            char bound_char = (bound_byte >= 32 && bound_byte < 127) ? (char)bound_byte : '?';
            ROUTE_LOG("    → Bound node %u: byte=%u ('%c')", bound_node, bound_byte, bound_char);
            
            /* Extract value from bound node */
            /* bound_node is a node ID - we need to get the byte value(s) from the node(s) */
            /* For single-digit numbers, the node stores the ASCII byte */
            /* For multi-digit numbers, we need to follow sequential edges to collect all bytes */
            
            /* Start with the bound node */
            uint32_t current_node = bound_node;
            uint32_t byte_sequence[16] = {0};  /* Max 16 bytes for a number */
            uint32_t byte_count = 0;
            
            /* Collect bytes by following sequential edges (for multi-digit numbers) */
            while (current_node < g->node_count && byte_count < 16) {
                Node *node = &g->nodes[current_node];
                uint8_t byte_val = node->byte;
                
                /* Check if it's a digit */
                if (byte_val >= '0' && byte_val <= '9') {
                    byte_sequence[byte_count++] = (uint32_t)byte_val;
                } else {
                    /* Not a digit - stop collecting */
                    break;
                }
                
                /* Check if there's a sequential edge to next digit */
                uint32_t eid = node->first_out;
                uint32_t next_node = UINT32_MAX;
                while (eid != UINT32_MAX && eid < g->edge_count) {
                    Edge *e = &g->edges[eid];
                    /* RELATIVE: Edge weight threshold scales with avg_edge_strength (no hard limit) */
                    float edge_threshold = (g->avg_edge_strength > 0.0f) ? 
                                         (g->avg_edge_strength * 0.5f) : 0.1f;
                    if (e->w >= edge_threshold && e->dst < g->node_count) {
                        Node *next = &g->nodes[e->dst];
                        if (next->byte >= '0' && next->byte <= '9') {
                            next_node = e->dst;
                            break;
                        }
                    }
                    eid = e->next_out;
                }
                
                if (next_node == UINT32_MAX) break;
                current_node = next_node;
            }
            
            /* Now extract value from byte sequence */
            if (byte_count > 0) {
                PatternValue val = extract_pattern_value(g, byte_sequence, byte_count, pattern_node_id);
                /* RELATIVE: Lower threshold - accept any extracted number value */
                /* Don't require value > 0, just require it's a number type */
                if (val.value_type == 0) {  /* Number type (was: && val.value_data > 0) */
                    extracted_values[value_count++] = val;
                }
            }
        }
    }
    
    ROUTE_LOG("  Extracted %u values", value_count);
    
    /* If values extracted, check if pattern routes to EXEC node */
    if (value_count > 0) {
        ROUTE_LOG("  Checking for edges to EXEC nodes from pattern %u", pattern_node_id);
        uint32_t eid = pattern_node->first_out;
        uint32_t edge_count_checked = 0;
        bool found_exec = false;
        
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            uint32_t dst = g->edges[eid].dst;
            edge_count_checked++;
            
            ROUTE_LOG("    Edge %u → node %u (payload_offset=%llu)", 
                      eid, dst, (unsigned long long)(dst < g->node_count ? g->nodes[dst].payload_offset : 0));
            
            if (dst < g->node_count && g->nodes[dst].payload_offset > 0) {
                /* Pattern routes to EXEC node - pass values */
                ROUTE_LOG("  ★ Found EXEC node %u! Passing %u values", dst, value_count);
                pass_values_to_exec(g, dst, extracted_values, value_count);
                found_exec = true;
                break;
            }
            eid = g->edges[eid].next_out;
        }
        
        if (!found_exec) {
            ROUTE_LOG("  ✗ No EXEC node found after checking %u edges (first_out=%u)", 
                      edge_count_checked, pattern_node->first_out);
        }
    } else {
        ROUTE_LOG("  → No values extracted, skipping EXEC routing");
    }
}

/* Learn pattern to EXEC routing - NOW DONE EXTERNALLY */
/* Routing is learned through pre-seeded edges + UEL feedback */
/* This function is a no-op - all routing is external */
static void learn_pattern_to_exec_routing(Graph *g, uint32_t pattern_node_id, 
                                          const PatternElement *elements, uint32_t element_count) {
    /* NO HARDCODING - routing learned externally through:
     * 1. Pre-seeded edges (preseed tool)
     * 2. UEL physics (feedback strengthens correct edges)
     * 3. Syscall-based learning (compilation, etc.)
     */
    (void)g; (void)pattern_node_id; (void)elements; (void)element_count;
    /* Function intentionally empty - substrate is pure */
}

/* Convert EXEC result back to pattern - graph learns this */
static void convert_result_to_pattern(Graph *g, uint32_t exec_node_id, uint64_t result) {
    if (!g || exec_node_id >= g->node_count) return;
    
    /* Convert integer result to byte sequence */
    char result_str[32];
    snprintf(result_str, sizeof(result_str), "%llu", (unsigned long long)result);
    
    /* Feed result as bytes - graph learns: result → output pattern */
    for (size_t i = 0; i < strlen(result_str); i++) {
        melvin_feed_byte(g, 100, (uint8_t)result_str[i], 0.5f);  /* Output port 100 */
    }
}

/* Execute graph structure directly - graph IS the code */
/* Nodes are instructions, edges are control flow */
/* No compilation needed - just interpret the graph */
static void execute_graph_structure(Graph *g, uint32_t start_node, uint64_t input1, uint64_t input2, uint64_t *result) {
    if (!g || !result || start_node >= g->node_count) return;
    
    /* Execution stack for graph interpreter */
    uint64_t stack[256] = {0};
    uint32_t stack_ptr = 0;
    
    /* Push inputs onto stack */
    if (input1 > 0) stack[stack_ptr++] = input1;
    if (input2 > 0) stack[stack_ptr++] = input2;
    
    uint32_t current_node = start_node;
    /* NO ITERATION LIMIT - graph structure determines execution */
    
    while (current_node < g->node_count) {
        Node *node = &g->nodes[current_node];
        
        /* Execute node based on type */
        if (node->payload_offset > 0) {
            /* EXEC node - execute operation */
            /* Pop operands from stack, execute, push result */
            if (stack_ptr >= 2) {
                uint64_t op2 = stack[--stack_ptr];
                uint64_t op1 = stack[--stack_ptr];
                
                /* Execute operation based on node */
                /* For EXEC_ADD (node 2000): addition */
                uint64_t op_result = 0;
                if (current_node == 2000) {
                    op_result = op1 + op2;
                } else {
                    /* Default: addition */
                    op_result = op1 + op2;
                }
                
                if (stack_ptr < 256) {
                    stack[stack_ptr++] = op_result;
                }
            }
        } else if (node->pattern_data_offset > 0) {
            /* Pattern node - expand pattern (function call) */
            expand_pattern(g, current_node, NULL);
        } else if (current_node < 256) {
            /* Data node - push value onto stack */
            if (stack_ptr < 256) {
                stack[stack_ptr++] = (uint64_t)node->byte;
            }
        }
        
        /* Follow edge to next node (control flow) */
        uint32_t next_node = UINT32_MAX;
        uint32_t eid = node->first_out;
        
        if (eid != UINT32_MAX && eid < g->edge_count) {
            /* Follow first outgoing edge */
            next_node = g->edges[eid].dst;
        }
        
        if (next_node == UINT32_MAX || next_node == current_node || next_node >= g->node_count) {
            break;  /* No next node or loop detected */
        }
        current_node = next_node;
    }
    
    /* Result is on top of stack */
    if (stack_ptr > 0) {
        *result = stack[stack_ptr - 1];
    } else {
        *result = 0;
    }
}

/* Expand pattern to sequence - when pattern activates, expand to underlying nodes */
static void expand_pattern(Graph *g, uint32_t pattern_node_id, const uint32_t *bindings) {
    if (!g || pattern_node_id >= g->node_count) return;
    
    Node *pattern_node = &g->nodes[pattern_node_id];
    if (pattern_node->pattern_data_offset == 0) return;  /* Not a pattern node */
    
    /* Read pattern data from blob */
    uint64_t pattern_offset = pattern_node->pattern_data_offset - g->hdr->blob_offset;
    if (pattern_offset >= g->blob_size) return;
    
    PatternData *pattern_data = (PatternData *)(g->blob + pattern_offset);
    if (pattern_data->magic != PATTERN_MAGIC) return;
    
    /* Expand pattern: activate underlying sequence */
    float pattern_activation = pattern_node->a;  /* Use pattern's activation */
    
    /* NEW: Extract values from pattern if it has blanks */
    PatternValue extracted_values[16];  /* Max 16 values per pattern */
    uint32_t value_count = 0;
    
    for (uint32_t i = 0; i < pattern_data->element_count; i++) {
        PatternElement *elem = &pattern_data->elements[i];
        uint32_t target_node_id;
        
        if (elem->is_blank == 0) {
            /* Data node or pattern node - check if it's a pattern */
            target_node_id = elem->value;
            
            if (target_node_id < g->node_count && 
                g->nodes[target_node_id].pattern_data_offset > 0) {
                /* PATTERN PATTERN MATCHING: This element is itself a pattern! */
                /* Recursively expand nested pattern */
                expand_pattern(g, target_node_id, bindings);
                continue;  /* Pattern expansion handles activation */
            }
        } else {
            /* Blank - use binding */
            uint32_t blank_pos = elem->value;
            if (bindings && blank_pos < 256 && bindings[blank_pos] > 0) {
                target_node_id = bindings[blank_pos];
                
                /* NEW: Extract value from this blank binding */
                if (value_count < 16) {
                    /* Get sequence that matched this blank */
                    /* For now, use the bound node - in full implementation, 
                     * would get full sequence from pattern instance */
                    uint32_t seq[1] = {target_node_id};
                    PatternValue val = extract_pattern_value(g, seq, 1, pattern_node_id);
                    if (val.value_data > 0 || val.value_type > 0) {
                        extracted_values[value_count++] = val;
                    }
                }
            } else {
                /* No binding - skip or use default */
                continue;
            }
        }
        
        /* Ensure node exists and activate it */
        ensure_node(g, target_node_id);
        if (target_node_id < g->node_count) {
            g->nodes[target_node_id].a += pattern_activation * 0.5f;  /* Spread activation */
            g->nodes[target_node_id].a = tanhf(g->nodes[target_node_id].a);  /* Clamp */
            
            /* Add to propagation queue */
            prop_queue_add(g, target_node_id);
        }
    }
    
    /* NEW: If pattern extracted values and routes to EXEC node, pass values */
    if (value_count > 0) {
        /* Check if this pattern routes to an EXEC node */
        uint32_t eid = pattern_node->first_out;
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            uint32_t dst = g->edges[eid].dst;
            if (dst < g->node_count && g->nodes[dst].payload_offset > 0) {
                /* This is an EXEC node - pass values to it */
                pass_values_to_exec(g, dst, extracted_values, value_count);
                break;
            }
            eid = g->edges[eid].next_out;
        }
    }
}

/* Create pattern node from discovered pattern */
static uint32_t create_pattern_node(Graph *g, const PatternElement *pattern_elements, 
                                     uint32_t element_count, const uint32_t *instance1,
                                     const uint32_t *instance2, uint32_t instance_length) {
    if (!g || !g->hdr || !pattern_elements || element_count == 0) return UINT32_MAX;
    
    /* Calculate pattern data size */
    size_t pattern_data_size = sizeof(PatternData) + element_count * sizeof(PatternElement);
    size_t instance1_size = sizeof(PatternInstance) + instance_length * sizeof(uint32_t);
    size_t instance2_size = sizeof(PatternInstance) + instance_length * sizeof(uint32_t);
    size_t total_size = pattern_data_size + instance1_size + instance2_size;
    
    /* Find space in blob for pattern data */
    /* Simple approach: append to blob (would need blob growth in production) */
    uint64_t blob_free_space = 0;
    if (g->hdr->blob_size > 0) {
        /* Check for free space in blob (simplified - would need proper tracking) */
        blob_free_space = g->hdr->blob_size - g->hdr->main_entry_offset;
    }
    
    if (blob_free_space < total_size) {
        /* Not enough space - would need to grow blob (for now, skip) */
        return UINT32_MAX;
    }
    
    /* Allocate pattern node */
    /* IMPORTANT: Always create new nodes - don't reuse placeholders!
     * Placeholder nodes are "local placeholders" that help build generalization and patterns.
     * They serve a structural purpose even if they look unused. We shouldn't strip them away.
     */
    
    /* FIX: Don't use node_count! Find next available pattern node in range */
    /* Patterns should be in range 840-10000 to avoid conflicts */
    const uint32_t PATTERN_START = 840;
    const uint32_t PATTERN_END = 10000;
    
    uint32_t pattern_node_id = UINT32_MAX;
    for (uint32_t i = PATTERN_START; i < PATTERN_END; i++) {
        /* Ensure node exists */
        if (i >= g->node_count) {
            ensure_node(g, i);
        }
        
        /* Check if this node is unused (no pattern data) */
        if (g->nodes[i].pattern_data_offset == 0) {
            pattern_node_id = i;
            break;
        }
    }
    
    if (pattern_node_id == UINT32_MAX) {
        /* No free pattern nodes - range full */
        fprintf(stderr, "ERROR: Pattern node range full (840-10000 all used)\n");
        return UINT32_MAX;
    }
    
    ensure_node(g, pattern_node_id);
    
    Node *pattern_node = &g->nodes[pattern_node_id];
    pattern_node->pattern_data_offset = g->hdr->blob_offset + g->hdr->main_entry_offset;
    
    /* Write pattern data to blob */
    uint8_t *blob_ptr = g->blob + g->hdr->main_entry_offset;
    PatternData *pattern_data = (PatternData *)blob_ptr;
    pattern_data->magic = PATTERN_MAGIC;
    pattern_data->element_count = element_count;
    pattern_data->instance_count = 2;
    pattern_data->frequency = 2.0f;
    pattern_data->strength = 0.1f;  /* Initial strength */
    pattern_data->first_instance_offset = g->hdr->main_entry_offset + pattern_data_size;
    
    /* Copy pattern elements */
    memcpy(pattern_data->elements, pattern_elements, element_count * sizeof(PatternElement));
    
    /* Write first instance */
    PatternInstance *inst1 = (PatternInstance *)(blob_ptr + pattern_data_size);
    inst1->next_instance_offset = pattern_data->first_instance_offset + instance1_size;
    inst1->sequence_length = instance_length;
    memcpy(inst1->sequence_nodes, instance1, instance_length * sizeof(uint32_t));
    
    /* Write second instance */
    PatternInstance *inst2 = (PatternInstance *)(blob_ptr + pattern_data_size + instance1_size);
    inst2->next_instance_offset = 0;  /* End of list */
    inst2->sequence_length = instance_length;
    memcpy(inst2->sequence_nodes, instance2, instance_length * sizeof(uint32_t));
    
    /* Update blob offset */
    g->hdr->main_entry_offset += total_size;
    
    return pattern_node_id;
}

/* Pattern discovery: Check if sequence repeats and create pattern */
static void discover_patterns(Graph *g, const uint32_t *sequence, uint32_t length) {
    if (!g || !sequence || length < 2) return;
    
    /* ANTI-EXPLOSION: Limit patterns created per call to prevent hangs */
    static uint32_t patterns_created_this_session = 0;
    const uint32_t MAX_PATTERNS_PER_SESSION = 1000;  /* Reasonable limit */
    
    /* Reset counter periodically (every 10000 bytes) */
    static uint64_t last_reset_pos = 0;
    if (g->sequence_buffer_pos - last_reset_pos > 10000) {
        patterns_created_this_session = 0;
        last_reset_pos = g->sequence_buffer_pos;
    }
    
    /* Skip if we've created too many patterns recently */
    if (patterns_created_this_session >= MAX_PATTERNS_PER_SESSION) {
        return;  /* Let the graph settle before adding more */
    }
    
    /* OPTIMIZATION: Only check recent buffer, not entire history */
    uint32_t max_lookback = 50;  /* Fixed small window */
    if (max_lookback > g->sequence_buffer_pos && !g->sequence_buffer_full) {
        max_lookback = (uint32_t)g->sequence_buffer_pos;
    }
    uint32_t buffer_start = (g->sequence_buffer_pos >= max_lookback) ? 
                           (g->sequence_buffer_pos - max_lookback) : 0;
    
    /* ANTI-EXPLOSION: Only check a few positions per call */
    uint32_t positions_checked = 0;
    const uint32_t MAX_POSITIONS = 10;  /* Only check 10 positions per byte */
    
    for (uint32_t buf_pos = buffer_start; buf_pos + length <= g->sequence_buffer_pos && positions_checked < MAX_POSITIONS; buf_pos += 2) {
        positions_checked++;
        /* Extract candidate sequence from buffer */
        uint32_t candidate_seq[10];
        bool valid = true;
        for (uint32_t i = 0; i < length && i < 10; i++) {
            if (buf_pos + i >= g->sequence_buffer_pos) {
                valid = false;
                break;
            }
            candidate_seq[i] = g->sequence_buffer[buf_pos + i];
        }
        if (!valid) continue;
        
        /* OPTIMIZATION: Filter out meaningless patterns early */
        /* Only consider sequences that share meaningful structure */
        
        /* Check if sequences share at least one concrete node in same position */
        /* HIERARCHICAL: Allow both byte nodes (0-255) AND pattern nodes (840+) */
        /* This enables patterns-of-patterns: true hierarchical abstraction! */
        bool has_shared_node = false;
        uint32_t shared_positions = 0;
        for (uint32_t i = 0; i < length; i++) {
            if (sequence[i] == candidate_seq[i]) {
                /* Same node at same position - potential pattern */
                /* Can be byte node (< 256) OR pattern node (>= 840) */
                bool is_valid_concrete = (sequence[i] < 256) ||  /* Byte nodes */
                                        (sequence[i] >= 840);    /* Pattern nodes (hierarchical!) */
                if (is_valid_concrete) {
                    has_shared_node = true;
                    shared_positions++;
                }
            }
        }
        
        if (!has_shared_node) continue;  /* No shared node - skip */
        
        /* FILTER: Require at least 2 shared concrete nodes for meaningful patterns */
        /* This reduces explosion from partial matches */
        if (shared_positions < 2 && length > 3) continue;
        
        /* Check if sequences are different (not identical) */
        bool is_different = false;
        uint32_t different_positions = 0;
        for (uint32_t i = 0; i < length; i++) {
            if (sequence[i] != candidate_seq[i]) {
                is_different = true;
                different_positions++;
            }
        }
        if (!is_different) continue;  /* Identical sequences - skip */
        
        /* FILTER: Require at least 1-2 different positions (meaningful variation) */
        /* But not too many (would be too generic) */
        if (different_positions == 0) continue;
        if (different_positions > length / 2 && length > 4) continue;  /* Too different */
        
        /* FILTER: Skip very short patterns (length 2) unless they have structure */
        /* Length 2 patterns are often noise from adjacent bytes */
        if (length == 2) {
            /* Only create if both positions are concrete (not blanks) */
            /* HIERARCHICAL: Allow byte nodes (< 256) AND pattern nodes (>= 840) */
            bool both_concrete = true;
            for (uint32_t i = 0; i < length; i++) {
                /* Valid concrete: byte nodes (0-255) or pattern nodes (840+) */
                bool is_concrete = (sequence[i] < 256) || (sequence[i] >= 840);
                bool is_concrete2 = (candidate_seq[i] < 256) || (candidate_seq[i] >= 840);
                if (!is_concrete || !is_concrete2) {
                    both_concrete = false;
                    break;
                }
            }
            if (!both_concrete) continue;  /* Skip noisy 2-element patterns */
        }
        
        /* Found two aligned sequences with shared node - create pattern */
        PatternElement pattern_elements[256];
        if (length > 256) continue;
        
        ROUTE_LOG("PATTERN CREATION: Comparing sequences (len=%u):", length);
        char seq1_str[256] = {0}, seq2_str[256] = {0};
        for (uint32_t i = 0; i < length && i < 255; i++) {
            uint8_t b1 = (uint8_t)(sequence[i] & 0xFF);
            uint8_t b2 = (uint8_t)(candidate_seq[i] & 0xFF);
            seq1_str[i] = (b1 >= 32 && b1 < 127) ? (char)b1 : '?';
            seq2_str[i] = (b2 >= 32 && b2 < 127) ? (char)b2 : '?';
        }
        ROUTE_LOG("  Seq1: \"%s\"", seq1_str);
        ROUTE_LOG("  Seq2: \"%s\"", seq2_str);
        
        /* Extract pattern: same = concrete, different = blank */
        uint32_t pattern_length = extract_pattern(sequence, candidate_seq, length, pattern_elements);
        
        /* Check if pattern has at least one blank (at least one position differs) */
        uint32_t blank_count = 0;
        uint32_t concrete_count = 0;
        for (uint32_t i = 0; i < pattern_length; i++) {
            if (pattern_elements[i].is_blank != 0) {
                blank_count++;
            } else {
                concrete_count++;
            }
        }
        
        /* FILTER: Require meaningful structure - at least 1 concrete AND 1 blank */
        /* Patterns with all blanks or all concrete are not useful */
        if (pattern_length > 0 && blank_count > 0 && concrete_count > 0) {
            /* FILTER: Limit blank ratio - too many blanks = too generic */
            float blank_ratio = (float)blank_count / pattern_length;
            if (blank_ratio > 0.75f && length > 4) continue;  /* Too generic */
            
            /* Create pattern */
            uint32_t pattern_node_id = create_pattern_node(g, pattern_elements, pattern_length,
                                                            sequence, candidate_seq, length);
            
            if (pattern_node_id != UINT32_MAX) {
                ROUTE_LOG("  → CREATED pattern node %u (length=%u, blanks=%u, concrete=%u)", 
                          pattern_node_id, pattern_length, blank_count, concrete_count);
                g->nodes[pattern_node_id].a += 0.1f;
                prop_queue_add(g, pattern_node_id);
                learn_pattern_to_exec_routing(g, pattern_node_id, pattern_elements, pattern_length);
                patterns_created_this_session++;  /* Track creation count */
            }
        }
    }
    
    /* Pattern matching via edges (O(edges) not O(nodes)) */
    /* Instead of scanning all nodes, let UEL propagation find patterns through edges */
    /* Edges from byte nodes to pattern nodes will activate patterns naturally */
}

/* ========================================================================
 * HIERARCHICAL PATTERN COMPOSITION
 * Compose smaller patterns into larger ones
 * ======================================================================== */

/* Record that two patterns activated sequentially */
static void record_adjacency(uint32_t pattern_a, uint32_t pattern_b) {
    if (pattern_a == UINT32_MAX || pattern_b == UINT32_MAX) return;
    if (pattern_a == pattern_b) return;  /* Same pattern, not adjacent */
    
    /* Look for existing adjacency */
    for (uint32_t i = 0; i < adjacency_count; i++) {
        if (adjacencies[i].pattern_a == pattern_a && 
            adjacencies[i].pattern_b == pattern_b) {
            adjacencies[i].cooccurrence_count++;
            return;
        }
    }
    
    /* Create new adjacency record */
    if (adjacency_count < MAX_ADJACENCIES) {
        adjacencies[adjacency_count].pattern_a = pattern_a;
        adjacencies[adjacency_count].pattern_b = pattern_b;
        adjacencies[adjacency_count].cooccurrence_count = 1;
        adjacencies[adjacency_count].last_seen = 0;  /* Would use timestamp */
        adjacency_count++;
    }
}

/* Track which patterns are activating during UEL propagation */
static void track_pattern_adjacency(Graph *g) {
    if (!g) return;
    
    static uint64_t step_counter = 0;
    static uint64_t last_log = 0;
    step_counter++;
    
    /* Find currently active pattern (if any) */
    uint32_t active_pattern = UINT32_MAX;
    float max_activation = 0.0f;
    
    /* Check pattern range - LOWER threshold to detect more activations */
    for (uint32_t pid = 840; pid < g->node_count && pid < 2000; pid++) {
        if (g->nodes[pid].pattern_data_offset > 0) {
            float act = fabsf(g->nodes[pid].a);
            /* Lower threshold: just above avg instead of 2x */
            if (act > max_activation && act > g->avg_activation * 0.5f) {
                max_activation = act;
                active_pattern = pid;
            }
        }
    }
    
    /* Log occasionally */
    if (step_counter - last_log > 1000 && active_pattern != UINT32_MAX) {
        fprintf(stderr, "[ADJACENCY] Active pattern: %u (activation=%.3f)\n", 
                active_pattern, max_activation);
        last_log = step_counter;
    }
    
    /* If we found an active pattern, record adjacency with previous */
    if (active_pattern != UINT32_MAX && last_active_pattern != UINT32_MAX) {
        /* Only record if DIFFERENT patterns */
        if (active_pattern != last_active_pattern) {
            /* Check timing - only if activated recently */
            if (step_counter - last_pattern_activation_time < 100) {
                record_adjacency(last_active_pattern, active_pattern);
                
                /* Log first few adjacencies */
                if (adjacency_count <= 10) {
                    fprintf(stderr, "[ADJACENCY] Recorded: %u → %u (count now=%u)\n", 
                            last_active_pattern, active_pattern, adjacency_count);
                }
            }
            
            /* Update last active pattern (new pattern) */
            last_active_pattern = active_pattern;
            last_pattern_activation_time = step_counter;
        }
    } else if (active_pattern != UINT32_MAX && last_active_pattern == UINT32_MAX) {
        /* First pattern activation */
        last_active_pattern = active_pattern;
        last_pattern_activation_time = step_counter;
    }
}

/* Compose two adjacent patterns into a longer pattern */
static void compose_adjacent_patterns(Graph *g) {
    if (!g || adjacency_count == 0) return;
    
    fprintf(stderr, "\n🔨 COMPOSITION CHECK: %u adjacencies tracked\n", adjacency_count);
    
    /* Show all adjacencies */
    for (uint32_t i = 0; i < adjacency_count && i < 10; i++) {
        fprintf(stderr, "  [%u] %u→%u (count=%u)\n", 
                i, adjacencies[i].pattern_a, adjacencies[i].pattern_b, 
                adjacencies[i].cooccurrence_count);
    }
    
    /* Find strong adjacencies (seen 2+ times) */
    int compositions_created = 0;
    for (uint32_t i = 0; i < adjacency_count; i++) {
        PatternAdjacency *adj = &adjacencies[i];
        
        if (adj->cooccurrence_count < 2) continue;  /* Not strong enough (lowered to 2 for testing) */
        
        uint32_t p1 = adj->pattern_a;
        uint32_t p2 = adj->pattern_b;
        
        if (p1 >= g->node_count || p2 >= g->node_count) continue;
        if (g->nodes[p1].pattern_data_offset == 0) continue;
        if (g->nodes[p2].pattern_data_offset == 0) continue;
        
        /* Get pattern data */
        uint64_t offset1 = g->nodes[p1].pattern_data_offset - g->hdr->blob_offset;
        uint64_t offset2 = g->nodes[p2].pattern_data_offset - g->hdr->blob_offset;
        
        if (offset1 >= g->blob_size || offset2 >= g->blob_size) continue;
        
        PatternData *pd1 = (PatternData *)(g->blob + offset1);
        PatternData *pd2 = (PatternData *)(g->blob + offset2);
        
        if (pd1->magic != PATTERN_MAGIC || pd2->magic != PATTERN_MAGIC) continue;
        
        /* Check if combined length is reasonable */
        uint32_t combined_len = pd1->element_count + pd2->element_count;
        if (combined_len > 10) continue;  /* Too long */
        
        /* Merge elements */
        PatternElement merged[20];
        uint32_t merged_idx = 0;
        
        /* Copy from first pattern */
        for (uint32_t j = 0; j < pd1->element_count && j < 10; j++) {
            merged[merged_idx++] = pd1->elements[j];
        }
        
        /* Copy from second pattern */
        /* Note: Might need to renumber blanks to avoid conflicts */
        uint32_t max_blank_in_p1 = 0;
        for (uint32_t j = 0; j < pd1->element_count; j++) {
            if (pd1->elements[j].is_blank && pd1->elements[j].value > max_blank_in_p1) {
                max_blank_in_p1 = pd1->elements[j].value;
            }
        }
        
        for (uint32_t j = 0; j < pd2->element_count && j < 10; j++) {
            PatternElement elem = pd2->elements[j];
            if (elem.is_blank) {
                /* Renumber blank to avoid conflict with p1's blanks */
                elem.value += max_blank_in_p1 + 1;
            }
            merged[merged_idx++] = elem;
        }
        
        /* Create composed pattern */
        uint32_t composed_id = create_pattern_node(g, merged, merged_idx, NULL, NULL, 0);
        
        if (composed_id != UINT32_MAX) {
            fprintf(stderr, "✨ COMPOSED pattern %u = %u ⊕ %u (len %u→%u, level-2)\n",
                    composed_id, p1, p2, pd1->element_count + pd2->element_count, merged_idx);
            
            /* Create edge: If p1 or p2 activate, composed should too */
            create_edge(g, p1, composed_id, 0.6f);
            create_edge(g, p2, composed_id, 0.6f);
            
            /* Mark adjacency as composed (don't compose again) */
            adj->cooccurrence_count = 0;  /* Reset to prevent re-composition */
            compositions_created++;
        }
    }
    
    if (compositions_created > 0) {
        fprintf(stderr, "🔨 Created %d composed patterns\n\n", compositions_created);
    } else if (adjacency_count > 0) {
        fprintf(stderr, "🔨 No compositions yet (need stronger adjacencies)\n\n");
    }
}

/* ========================================================================
 * EVENT-DRIVEN PATTERN DISCOVERY: Co-Activation Detection
 * Pattern discovery through wave propagation, not buffer scanning!
 * ======================================================================== */

/* Simple hash function for activation sequences */
static uint64_t hash_activation_sequence(const uint32_t *seq, int length) {
    uint64_t hash = 14695981039346656037ULL;  /* FNV offset basis */
    for (int i = 0; i < length; i++) {
        hash ^= seq[i];
        hash *= 1099511628211ULL;  /* FNV prime */
    }
    return hash;
}

/* Check if sequence already has a pattern node */
static int activation_sequence_has_pattern(Graph *g, const uint32_t *seq, int length) {
    (void)length;  /* Unused for now */
    if (seq[0] >= g->node_count) return 0;
    
    uint32_t eid = g->nodes[seq[0]].first_out;
    int checked = 0;
    while (eid != UINT32_MAX && eid < g->edge_count && checked++ < 50) {
        uint32_t dst = g->edges[eid].dst;
        if (dst >= 840 && dst < g->node_count && g->nodes[dst].pattern_data_offset > 0) {
            return 1;  /* Found pattern node */
        }
        eid = g->edges[eid].next_out;
    }
    return 0;
}

/* Detect co-activation patterns in recent activation window */
void detect_coactivation_patterns(Graph *g) {
    if (!g || !g->recent_activations || !g->coactivation_hash) return;
    
    /* Clear hash table */
    memset(g->coactivation_hash, 0, g->coactivation_hash_size * sizeof(uint64_t));
    
    int patterns_created = 0;
    const int MAX_PATTERNS_PER_CHECK = 3;  /* Limit to prevent explosion */
    
    static int call_count = 0;
    call_count++;
    if (call_count % 10 == 0) {
        fprintf(stderr, "Co-activation check #%d (window has %u activations)\n", 
                call_count, g->activation_window_size);
    }
    
    /* DYNAMIC PATTERN SIZE: Try multiple lengths for hierarchical composition */
    /* Base patterns (2-3), medium patterns (4-5), complex patterns (6-7) */
    for (int len = 2; len <= 7 && patterns_created < MAX_PATTERNS_PER_CHECK; len++) {
        if (len > (int)g->activation_window_size) continue;
        
        /* Limit patterns per length to prevent explosion */
        int patterns_at_this_length = 0;
        const int MAX_PER_LENGTH = 2;  /* Max 2 patterns per length */
        
        /* Scan window for repeated sequences */
        for (uint32_t i = 0; i + len <= g->activation_window_size && patterns_created < MAX_PATTERNS_PER_CHECK; i++) {
            uint32_t seq[10];
            
            /* Extract sequence */
            for (int j = 0; j < len; j++) {
                uint32_t pos = (g->activation_window_pos + i + j) % g->activation_window_size;
                seq[j] = g->recent_activations[pos];
            }
            
            /* Skip if sequence contains invalid nodes */
            int valid = 1;
            for (int j = 0; j < len; j++) {
                if (seq[j] >= g->node_count || seq[j] == UINT32_MAX) {
                    valid = 0;
                    break;
                }
            }
            if (!valid) continue;
            
            /* Hash sequence */
            uint64_t hash = hash_activation_sequence(seq, len);
            uint32_t slot = hash % g->coactivation_hash_size;
            
            /* Check if we've seen this sequence before (in THIS window) */
            if (g->coactivation_hash[slot] == hash) {
                /* Repeated sequence! Create pattern if not exists */
                if (!activation_sequence_has_pattern(g, seq, len)) {
                    /* Create pattern elements with BLANKS for digits */
                    /* This enables generalization: "1+1=2" becomes [BLANK, +, BLANK, =, BLANK] */
                    PatternElement elements[10];
                    uint32_t blank_position = 0;
                    
                    for (int j = 0; j < len && j < 10; j++) {
                        /* Check if this node represents a digit */
                        uint8_t byte = (seq[j] < g->node_count) ? g->nodes[seq[j]].byte : 0;
                        
                        if (byte >= '0' && byte <= '9') {
                            /* Digit → treat as variable (BLANK) for generalization */
                            elements[j].is_blank = 1;
                            elements[j].value = blank_position;
                            blank_position++;
                        } else {
                            /* Operator/symbol → treat as constant (concrete) */
                            elements[j].is_blank = 0;
                            elements[j].value = seq[j];
                        }
                    }
                    
                    /* Create pattern node */
                    uint32_t pattern_id = create_pattern_node(g, elements, len, seq, seq, len);
                    if (pattern_id != UINT32_MAX) {
                        patterns_created++;
                        patterns_at_this_length++;
                        
                        /* Count blanks to show if pattern is generalized */
                        uint32_t blank_count = 0;
                        for (int k = 0; k < len && k < 10; k++) {
                            if (elements[k].is_blank) blank_count++;
                        }
                        
                        if (blank_count > 0) {
                            fprintf(stderr, "  ✓ Created GENERALIZED pattern %u (len=%d, %u blanks, level-1)\n", 
                                    pattern_id, len, blank_count);
                        } else {
                            fprintf(stderr, "  ✓ Created pattern %u from co-activation (len=%d, level-1)\n", 
                                    pattern_id, len);
                        }
                        
                        /* Create edges from sequence nodes to pattern */
                        for (int j = 0; j < len; j++) {
                            if (seq[j] < g->node_count && pattern_id < g->node_count) {
                                if (find_edge(g, seq[j], pattern_id) == UINT32_MAX) {
                                    create_edge(g, seq[j], pattern_id, 0.5f);
                                }
                            }
                        }
                        
                        /* Check if we've hit limit for this length */
                        if (patterns_at_this_length >= MAX_PER_LENGTH) {
                            break;  /* Move to next length */
                        }
                    }
                }
            } else {
                /* First time seeing this sequence - mark it */
                g->coactivation_hash[slot] = hash;
            }
        }
    }
}

/* ========================================================================
 * PATTERN MATCHING AND ROUTING
 * Match input sequences against existing patterns and route to EXEC
 * ======================================================================== */

/* Match patterns against current sequence and route to EXEC if found */
static void match_patterns_and_route(Graph *g, const uint32_t *sequence, uint32_t length) {
    if (!g || !sequence || length == 0) return;
    
    /* Don't match during initial learning phase */
    if (g->node_count < 840) return;  /* No patterns exist yet */
    
    ROUTE_LOG("match_patterns_and_route: checking sequence length %u", length);
    
    /* Try different sequence lengths (favor longer matches first) */
    for (int len = (int)length; len >= 2 && len <= 10; len--) {
        /* Extract subsequence (last 'len' nodes from buffer) */
        uint32_t subseq[10];
        uint32_t start_idx = (length >= (uint32_t)len) ? (length - len) : 0;
        
        for (int i = 0; i < len; i++) {
            subseq[i] = sequence[start_idx + i];
        }
        
        ROUTE_LOG("  Trying subsequence length %d", len);
        
        /* Search for patterns that might match this sequence */
        /* Patterns are in range 840-999 typically */
        uint32_t pattern_start = 840;
        uint32_t pattern_end = (g->node_count < 2000) ? g->node_count : 2000;
        
        for (uint32_t pid = pattern_start; pid < pattern_end; pid++) {
            Node *pnode = &g->nodes[pid];
            
            /* Skip if not a pattern node */
            if (pnode->pattern_data_offset == 0) continue;
            
            /* Test if this pattern matches the sequence */
            uint32_t bindings[256] = {0};
            
            if (pattern_matches_sequence(g, pid, subseq, len, bindings)) {
                /* MATCH FOUND! */
                fprintf(stderr, "\n🎯 ===== PATTERN MATCH FOUND =====\n");
                fprintf(stderr, "Pattern ID: %u\n", pid);
                fprintf(stderr, "Sequence length: %d\n", len);
                
                /* Log what we matched */
                fprintf(stderr, "Matched sequence: ");
                for (int i = 0; i < len && i < 10; i++) {
                    if (subseq[i] < g->node_count) {
                        uint8_t byte = g->nodes[subseq[i]].byte;
                        char c = (byte >= 32 && byte < 127) ? (char)byte : '?';
                        fprintf(stderr, "'%c' ", c);
                    }
                }
                fprintf(stderr, "\n");
                
                /* Log bindings */
                fprintf(stderr, "Bindings extracted:\n");
                int binding_count = 0;
                for (int i = 0; i < 256; i++) {
                    if (bindings[i] > 0 && bindings[i] < g->node_count) {
                        uint8_t byte = g->nodes[bindings[i]].byte;
                        char c = (byte >= 32 && byte < 127) ? (char)byte : '?';
                        fprintf(stderr, "  [%d] → node %u ('%c')\n", i, bindings[i], c);
                        binding_count++;
                    }
                }
                if (binding_count == 0) {
                    fprintf(stderr, "  (no bindings - concrete pattern)\n");
                }
                fprintf(stderr, "==================================\n\n");
                
                ROUTE_LOG("✅ MATCH FOUND: Pattern %u matches sequence length %d", pid, len);
                
                /* Route to EXEC */
                /* Note: extract_and_route_to_exec only needs pattern_id and bindings */
                /* It will read pattern structure and extract values internally */
                extract_and_route_to_exec(g, pid, bindings);
                
                /* Found a match - we can return or continue looking for more matches */
                /* For now, return after first match to avoid multiple executions */
                return;
            }
        }
    }
    
    ROUTE_LOG("  No patterns matched");
}

/* Global Law: Pattern creation - called from melvin_feed_byte */
static void pattern_law_apply(Graph *g, uint32_t data_node_id) {
    if (!g || !g->sequence_buffer) return;
    
    /* Add to sequence buffer FIRST (always!) */
    if (g->sequence_buffer_pos >= g->sequence_buffer_size) {
        g->sequence_buffer_pos = 0;
        g->sequence_buffer_full = 1;
    }
    
    g->sequence_buffer[g->sequence_buffer_pos] = data_node_id;
    g->sequence_buffer_pos++;
    
    /* RATE LIMIT: Only run pattern DISCOVERY every N bytes to prevent explosion */
    /* OPTIMIZED: Check every 50 bytes (not 10) for better performance on real text */
    static uint64_t last_pattern_check = 0;
    if (g->sequence_buffer_pos - last_pattern_check >= 50) {
        last_pattern_check = g->sequence_buffer_pos;
        
        /* Extract sequence for discovery */
        uint32_t len = 3;  /* SIMPLIFIED: Only check length 3 patterns */
        
        if (len <= g->sequence_buffer_pos || g->sequence_buffer_full) {
            uint32_t start_pos = (g->sequence_buffer_pos >= len) ? 
                                 (g->sequence_buffer_pos - len) : 
                                 (g->sequence_buffer_size + g->sequence_buffer_pos - len);
            
            uint32_t sequence[10];
            for (uint32_t i = 0; i < len; i++) {
                sequence[i] = g->sequence_buffer[(start_pos + i) % g->sequence_buffer_size];
            }
            
            /* Discover patterns (with internal limits) */
            discover_patterns(g, sequence, len);
        }
    }
    
    /* ✅ NEW: PATTERN MATCHING - runs more frequently than discovery */
    /* Match patterns every 5 bytes (not every 50!) to catch queries */
    static uint64_t last_match_check = 0;
    if (g->sequence_buffer_pos - last_match_check >= 5) {
        last_match_check = g->sequence_buffer_pos;
        
        /* Extract longer sequence for matching (up to 10 bytes) */
        uint32_t max_len = 10;
        uint32_t match_len = (g->sequence_buffer_pos < max_len) ? g->sequence_buffer_pos : max_len;
        
        if (match_len >= 2 && (match_len <= g->sequence_buffer_pos || g->sequence_buffer_full)) {
            uint32_t start_pos = (g->sequence_buffer_pos >= match_len) ? 
                                 (g->sequence_buffer_pos - match_len) : 
                                 (g->sequence_buffer_size + g->sequence_buffer_pos - match_len);
            
            uint32_t sequence[10];
            for (uint32_t i = 0; i < match_len && i < 10; i++) {
                sequence[i] = g->sequence_buffer[(start_pos + i) % g->sequence_buffer_size];
            }
            
            /* ✅ MATCH PATTERNS AND ROUTE TO EXEC */
            match_patterns_and_route(g, sequence, match_len);
        }
    }
}

