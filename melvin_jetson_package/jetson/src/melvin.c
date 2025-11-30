/*
 * melvin.c - Binary Brain Loader + UEL Physics
 * 
 * Does:
 *   1. mmap .m file
 *   2. feed bytes (write to .m)
 *   3. expose syscalls
 *   4. run UEL physics (embedded in this file)
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

/* ========================================================================
 * SOFT STRUCTURE INITIALIZATION
 * ======================================================================== */

/* Forward declarations */
static void initialize_soft_structure(Graph *g, bool is_new_file);
static void create_initial_edge_suggestions(Graph *g, bool is_new_file);
static uint32_t find_edge(Graph *g, uint32_t src, uint32_t dst);
static uint32_t create_edge(Graph *g, uint32_t src, uint32_t dst, float w);

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
        /* TOOL GATEWAY PORTS (300-699): Pattern generation tools */
        else if (i < 700) {
            n->semantic_hint = 300;  /* Tool gateway range */
        }
        /* CODE PATTERN PORTS (700-799): Compiled code patterns (graph learns from machine code) */
        else if (i < 800) {
            n->semantic_hint = 700;  /* Code pattern range */
            n->memory_propensity = 0.9f;  /* High retention - code patterns are important */
            n->input_propensity = 0.6f;  /* Receives compiled code */
            n->output_propensity = 0.4f;  /* Can produce code-like patterns */
            
            /* STT Gateway (300-319): Audio → Text */
            if (i >= 300 && i < 320) {
                if (i < 310) {
                    /* STT input (300-309) */
                    n->input_propensity = 0.7f;
                    n->output_propensity = 0.3f;
                } else {
                    /* STT output (310-319) */
                    n->input_propensity = 0.3f;
                    n->output_propensity = 0.8f;  /* High - produces text output */
                }
            }
            /* Vision Gateway (400-419): Image → Labels */
            else if (i >= 400 && i < 420) {
                if (i < 410) {
                    /* Vision input (400-409) */
                    n->input_propensity = 0.7f;
                    n->output_propensity = 0.3f;
                } else {
                    /* Vision output (410-419) */
                    n->input_propensity = 0.3f;
                    n->output_propensity = 0.8f;  /* High - produces labels */
                }
            }
            /* LLM Gateway (500-519): Text → Text */
            else if (i >= 500 && i < 520) {
                if (i < 510) {
                    /* LLM input (500-509) */
                    n->input_propensity = 0.7f;
                    n->output_propensity = 0.3f;
                } else {
                    /* LLM output (510-519) */
                    n->input_propensity = 0.3f;
                    n->output_propensity = 0.9f;  /* Very high - produces text output */
                }
            }
            /* TTS Gateway (600-619): Text → Audio */
            else if (i >= 600 && i < 620) {
                if (i < 610) {
                    /* TTS input (600-609) */
                    n->input_propensity = 0.7f;
                    n->output_propensity = 0.3f;
                } else {
                    /* TTS output (610-619) */
                    n->input_propensity = 0.3f;
                    n->output_propensity = 0.8f;  /* High - produces audio output */
                }
            }
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
    /* Create edges for ALL input ports, not just first 10 */
    for (uint32_t input = 0; input < 100 && input < g->node_count; input++) {
        for (uint32_t memory = 200; memory < 210 && memory < g->node_count; memory++) {
            if (find_edge(g, input, memory) == UINT32_MAX) {
                create_edge(g, input, memory, base_weight);  /* RELATIVE: Weak initial weight */
            }
        }
    }
    
    /* 2. Working Memory → Output (ports 200-209 → 100-199) */
    /* Create edges for ALL output ports, not just first 10 */
    for (uint32_t memory = 200; memory < 210 && memory < g->node_count; memory++) {
        for (uint32_t output = 100; output < 200 && output < g->node_count; output++) {
            if (find_edge(g, memory, output) == UINT32_MAX) {
                create_edge(g, memory, output, base_weight);  /* RELATIVE: Weak initial weight */
            }
        }
    }
    
    /* 3. Output → Feedback (ports 100-199 → 30-33) */
    /* Create edges for ALL output ports, not just first 10 */
    for (uint32_t output = 100; output < 200 && output < g->node_count; output++) {
        for (uint32_t feedback = 30; feedback < 34 && feedback < g->node_count; feedback++) {
            if (find_edge(g, output, feedback) == UINT32_MAX) {
                create_edge(g, output, feedback, weak_weight);  /* RELATIVE: Very weak - graph learns correlation */
            }
        }
    }
    
    /* 4. Sparse Memory → Memory connections (create more connections) */
    for (uint32_t i = 0; i < 50; i++) {  /* Increased from 20 to 50 */
        uint32_t mem1 = 200 + (i % 10);
        uint32_t mem2 = 200 + ((i * 7) % 10);  /* Pseudo-random pairing */
        if (mem1 != mem2 && mem1 < g->node_count && mem2 < g->node_count) {
            if (find_edge(g, mem1, mem2) == UINT32_MAX) {
                create_edge(g, mem1, mem2, memory_weight);  /* RELATIVE: Weak memory connections */
            }
        }
    }
    
    /* 5. Temporal anchor connections */
    if (g->node_count > 243) {
        /* "now" (240) → "recent" (241) */
        if (find_edge(g, 240, 241) == UINT32_MAX) {
            create_edge(g, 240, 241, temporal_medium);  /* RELATIVE: Slightly stronger - temporal flow */
        }
        /* "recent" (241) → "memory" (242) */
        if (find_edge(g, 241, 242) == UINT32_MAX) {
            create_edge(g, 241, 242, temporal_weak);  /* RELATIVE */
        }
        /* "memory" (242) → "future" (243) */
        if (find_edge(g, 242, 243) == UINT32_MAX) {
            create_edge(g, 242, 243, base_weight);  /* RELATIVE */
        }
    }
    
    /* 6. Hardware instinct patterns - bootstrap tool connections */
    /* These provide initial routing, graph can strengthen/weaken/rewire */
    
    /* Mic → Audio Processing → STT Gateway (300-309) */
    if (g->node_count > 300) {
        /* Mic port (0) → Working memory (200) */
        if (find_edge(g, 0, 200) == UINT32_MAX) {
            create_edge(g, 0, 200, base_weight);  /* RELATIVE: Mic → Working memory */
        }
        /* Working memory → STT gateway input (300) */
        if (find_edge(g, 200, 300) == UINT32_MAX) {
            create_edge(g, 200, 300, weak_weight);  /* RELATIVE: Weak - graph learns when to use STT */
        }
        /* STT gateway output (310) → Working memory */
        if (find_edge(g, 310, 201) == UINT32_MAX) {
            create_edge(g, 310, 201, base_weight);  /* RELATIVE: STT output → Working memory */
        }
        /* Working memory → Speaker port (100) */
        if (find_edge(g, 201, 100) == UINT32_MAX) {
            create_edge(g, 201, 100, weak_weight);  /* RELATIVE: Weak - graph learns routing */
        }
    }
    
    /* Camera → Vision Processing → Vision Gateway (400-409) */
    if (g->node_count > 400) {
        /* Camera port (10) → Working memory (201) */
        if (find_edge(g, 10, 201) == UINT32_MAX) {
            create_edge(g, 10, 201, base_weight);  /* RELATIVE: Camera → Working memory */
        }
        /* Working memory → Vision gateway input (400) */
        if (find_edge(g, 201, 400) == UINT32_MAX) {
            create_edge(g, 201, 400, weak_weight);  /* RELATIVE: Weak - graph learns when to use vision */
        }
        /* Vision gateway output (410) → Working memory */
        if (find_edge(g, 410, 202) == UINT32_MAX) {
            create_edge(g, 410, 202, base_weight);  /* RELATIVE: Vision output → Working memory */
        }
        /* Working memory → Display port (110) */
        if (find_edge(g, 202, 110) == UINT32_MAX) {
            create_edge(g, 202, 110, weak_weight);  /* RELATIVE: Weak - graph learns routing */
        }
    }
    
    /* Text → LLM Processing → LLM Gateway (500-509) */
    if (g->node_count > 500) {
        /* Text input port (20) → Working memory (202) */
        if (find_edge(g, 20, 202) == UINT32_MAX) {
            create_edge(g, 20, 202, base_weight);  /* RELATIVE: Text → Working memory */
        }
        /* Working memory → LLM gateway input (500) */
        if (find_edge(g, 202, 500) == UINT32_MAX) {
            create_edge(g, 202, 500, weak_weight);  /* RELATIVE: Weak - graph learns when to use LLM */
        }
        /* LLM gateway output (510) → Working memory */
        if (find_edge(g, 510, 203) == UINT32_MAX) {
            create_edge(g, 510, 203, base_weight);  /* RELATIVE: LLM output → Working memory */
        }
        /* Working memory → Text output port (100) or TTS gateway (600) */
        if (find_edge(g, 203, 100) == UINT32_MAX) {
            create_edge(g, 203, 100, weak_weight);  /* RELATIVE: Weak - graph learns routing */
        }
    }
    
    /* TTS Gateway (600-609) */
    if (g->node_count > 600) {
        /* Text → TTS gateway input (600) */
        if (find_edge(g, 203, 600) == UINT32_MAX) {
            create_edge(g, 203, 600, weak_weight);  /* RELATIVE: Weak - graph learns when to use TTS */
        }
        /* TTS gateway output (610) → Speaker port (100) */
        if (find_edge(g, 610, 100) == UINT32_MAX) {
            create_edge(g, 610, 100, base_weight);  /* RELATIVE: TTS output → Speaker */
        }
    }
    
    /* 7. Cross-tool connections (very weak - graph learns these patterns) */
    /* These enable hierarchical pattern building */
    if (g->node_count > 600) {
        /* STT ↔ Vision (audio-visual connection) */
        if (find_edge(g, 310, 410) == UINT32_MAX) {
            create_edge(g, 310, 410, very_weak_weight);  /* RELATIVE: Very weak - cross-tool pattern */
        }
        if (find_edge(g, 410, 310) == UINT32_MAX) {
            create_edge(g, 410, 310, very_weak_weight);  /* RELATIVE */
        }
        
        /* Vision ↔ LLM (visual-text connection) */
        if (find_edge(g, 410, 510) == UINT32_MAX) {
            create_edge(g, 410, 510, very_weak_weight);  /* RELATIVE: Very weak - cross-tool pattern */
        }
        if (find_edge(g, 510, 410) == UINT32_MAX) {
            create_edge(g, 510, 410, very_weak_weight);  /* RELATIVE */
        }
        
        /* LLM ↔ STT (text-audio connection) */
        if (find_edge(g, 510, 310) == UINT32_MAX) {
            create_edge(g, 510, 310, very_weak_weight);  /* RELATIVE: Very weak - cross-tool pattern */
        }
        if (find_edge(g, 310, 510) == UINT32_MAX) {
            create_edge(g, 310, 510, very_weak_weight);  /* RELATIVE */
        }
        
        /* LLM ↔ TTS (text generation → speech) */
        if (find_edge(g, 510, 600) == UINT32_MAX) {
            create_edge(g, 510, 600, weak_weight);  /* RELATIVE: Slightly stronger - common pattern */
        }
    }
    
    /* 8. ERROR HANDLING PATTERNS (graph learns from failures through UEL) */
    /* Error detection nodes: 250-259 */
    if (g->node_count > 259) {
        /* Tool failures → Error detection (250) */
        /* STT failure signal */
        if (find_edge(g, 300, 250) == UINT32_MAX) {
            create_edge(g, 300, 250, base_weight);  /* RELATIVE: Tool input → Error detection */
        }
        /* Vision failure signal */
        if (find_edge(g, 400, 250) == UINT32_MAX) {
            create_edge(g, 400, 250, base_weight);  /* RELATIVE */
        }
        /* LLM failure signal */
        if (find_edge(g, 500, 250) == UINT32_MAX) {
            create_edge(g, 500, 250, base_weight);  /* RELATIVE */
        }
        /* TTS failure signal */
        if (find_edge(g, 600, 250) == UINT32_MAX) {
            create_edge(g, 600, 250, base_weight);  /* RELATIVE */
        }
        
        /* Error detection → Recovery patterns (251-254) */
        for (uint32_t recovery = 251; recovery < 255 && recovery < g->node_count; recovery++) {
            if (find_edge(g, 250, recovery) == UINT32_MAX) {
                create_edge(g, 250, recovery, memory_weight);  /* RELATIVE: Error → Recovery */
            }
            /* Recovery → Retry tool or fallback */
            if (recovery == 251 && find_edge(g, 251, 300) == UINT32_MAX) {
                create_edge(g, 251, 300, weak_weight);  /* RELATIVE: Recovery → Retry STT */
            }
            if (recovery == 252 && find_edge(g, 252, 500) == UINT32_MAX) {
                create_edge(g, 252, 500, weak_weight);  /* RELATIVE: Recovery → Retry LLM */
            }
        }
        
        /* Error → Feedback (graph learns from errors) */
        if (find_edge(g, 250, 31) == UINT32_MAX) {
            create_edge(g, 250, 31, base_weight);  /* RELATIVE: Error → Negative feedback */
        }
    }
    
    /* 9. AUTOMATIC TOOL INTEGRATION PATTERNS */
    /* Graph learns when to call tools through pattern recognition */
    if (g->node_count > 600) {
        /* Input patterns → Tool gateway activation (graph learns thresholds) */
        /* Audio input (0) → STT gateway when pattern matches */
        for (uint32_t pattern = 200; pattern < 210 && pattern < g->node_count; pattern++) {
            if (find_edge(g, pattern, 300) == UINT32_MAX) {
                create_edge(g, pattern, 300, extra_weak_weight);  /* RELATIVE: Very weak - graph learns when */
            }
        }
        
        /* Text patterns → LLM gateway when needed */
        for (uint32_t pattern = 201; pattern < 211 && pattern < g->node_count; pattern++) {
            if (find_edge(g, pattern, 500) == UINT32_MAX) {
                create_edge(g, pattern, 500, extra_weak_weight);  /* RELATIVE: Very weak - graph learns when */
            }
        }
        
        /* Tool outputs → Automatic graph feeding (stronger - these create patterns) */
        /* STT output → Graph nodes (automatic pattern creation) */
        for (uint32_t mem = 200; mem < 210 && mem < g->node_count; mem++) {
            if (find_edge(g, 310, mem) == UINT32_MAX) {
                create_edge(g, 310, mem, medium_weight);  /* RELATIVE: Tool output → Graph (creates patterns) */
            }
        }
        
        /* LLM output → Graph nodes */
        for (uint32_t mem = 201; mem < 211 && mem < g->node_count; mem++) {
            if (find_edge(g, 510, mem) == UINT32_MAX) {
                create_edge(g, 510, mem, medium_weight);  /* RELATIVE: Tool output → Graph */
            }
        }
        
        /* Vision output → Graph nodes */
        for (uint32_t mem = 202; mem < 212 && mem < g->node_count; mem++) {
            if (find_edge(g, 410, mem) == UINT32_MAX) {
                create_edge(g, 410, mem, medium_weight);  /* RELATIVE: Tool output → Graph */
            }
        }
        
        /* TTS output → Graph nodes (audio patterns) */
        for (uint32_t mem = 203; mem < 213 && mem < g->node_count; mem++) {
            if (find_edge(g, 610, mem) == UINT32_MAX) {
                create_edge(g, 610, mem, medium_weight);  /* RELATIVE: Tool output → Graph */
            }
        }
    }
    
    /* 10. SELF-REGULATION PATTERNS (graph controls its own activity) */
    if (g->node_count > 260) {
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
        for (uint32_t output = 100; output < 110 && output < g->node_count; output++) {
            if (find_edge(g, output, 255) == UINT32_MAX) {
                create_edge(g, output, 255, weak_weight);  /* RELATIVE: Output → Chaos monitor */
            }
        }
    }
    
    /* 11. TOOL SUCCESS/FAILURE FEEDBACK LOOPS */
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
        if (ftruncate(fd, total_size) < 0) {
            close(fd);
            free(g);
            return NULL;
        }
        
        /* mmap */
        void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            close(fd);
            free(g);
            return NULL;
        }
        
        /* Initialize header */
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
        memset((char *)map + hdr->nodes_offset, 0, (size_t)nodes_size);
        memset((char *)map + hdr->edges_offset, 0, (size_t)edges_size);
        memset((char *)map + hdr->blob_offset, 0, (size_t)blob_size_u64);
        /* Cold data left as-is (will be filled by corpus loader) */
        
        /* Initialize data nodes (0-255) - just structure, no physics */
        Node *nodes = (Node *)((char *)map + hdr->nodes_offset);
        for (int i = 0; i < 256 && i < (int)initial_nodes; i++) {
            nodes[i].byte = (uint8_t)i;
            nodes[i].a = 0.0f;
            nodes[i].first_in = UINT32_MAX;
            nodes[i].first_out = UINT32_MAX;
            nodes[i].in_degree = 0;
            nodes[i].out_degree = 0;
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
        
        /* Initialize UEL physics state (always initialize, even for existing files) */
        if (g->node_count > 0) {
            /* Calculate initial estimates from existing graph */
            float sum_activation = 0.0f;
            float sum_edge_strength = 0.0f;
            uint64_t active_count = 0;
            
            for (uint64_t i = 0; i < g->node_count; i++) {
                float a_abs = fabsf(g->nodes[i].a);
                if (a_abs > 0.001f) {
                    sum_activation += a_abs;
                    active_count++;
                }
            }
            
            for (uint64_t i = 0; i < g->edge_count; i++) {
                sum_edge_strength += fabsf(g->edges[i].w);
            }
            
            g->avg_activation = (active_count > 0) ? (sum_activation / active_count) : 0.1f;
            if (g->avg_activation < 0.01f) g->avg_activation = 0.1f;  /* Minimum */
            
            g->avg_edge_strength = (g->edge_count > 0) ? (sum_edge_strength / g->edge_count) : 0.1f;
            if (g->avg_edge_strength < 0.01f) g->avg_edge_strength = 0.1f;  /* Minimum */
            
            /* Estimate initial chaos from activation variance */
            float variance = 0.0f;
            for (uint64_t i = 0; i < g->node_count && i < 100; i++) {  /* Sample first 100 */
                float diff = fabsf(g->nodes[i].a) - g->avg_activation;
                variance += diff * diff;
            }
            g->avg_chaos = (g->node_count > 0) ? sqrtf(variance / (float)(g->node_count < 100 ? g->node_count : 100)) : 0.1f;
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
        
        /* Initialize propagation queue for new files */
        g->prop_queue_size = (g->node_count > 0) ? g->node_count : 256;
        g->prop_queue = calloc(g->prop_queue_size, sizeof(uint32_t));
        g->prop_queued = calloc(g->prop_queue_size, sizeof(_Atomic uint8_t));
        if (g->prop_queue && g->prop_queued) {
            g->prop_queue_head = 0;
            g->prop_queue_tail = 0;
            for (uint64_t i = 0; i < g->prop_queue_size; i++) {
                g->prop_queued[i] = 0;
            }
        }
        
        /* Initialize tracking arrays for new files */
        g->last_activation = calloc(g->node_count, sizeof(float));
        g->last_message = calloc(g->node_count, sizeof(float));
        g->output_propensity = calloc(g->node_count, sizeof(float));
        g->feedback_correlation = calloc(g->node_count, sizeof(float));
        g->prediction_accuracy = calloc(g->node_count, sizeof(float));
        if (g->last_activation && g->last_message && g->output_propensity && 
            g->feedback_correlation && g->prediction_accuracy) {
            /* Initialize from current state */
            for (uint64_t i = 0; i < g->node_count; i++) {
                g->last_activation[i] = g->nodes[i].a;
                /* Initialize output_propensity from node's output_propensity field */
                g->output_propensity[i] = g->nodes[i].output_propensity;
            }
        }
        
        /* Initialize soft structure scaffolding (embedded in .m file on bootup) */
        initialize_soft_structure(g, true);  /* is_new_file = true */
        
        /* Create weak initial edge suggestions (graph can strengthen/weaken/rewire) */
        create_initial_edge_suggestions(g, true);  /* is_new_file = true */
        
        
    } else {
        /* Open existing file */
        void *map = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
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
        if (g->node_count > 0) {
            /* Calculate initial estimates from existing graph */
            float sum_activation = 0.0f;
            float sum_edge_strength = 0.0f;
            uint64_t active_count = 0;
            
            for (uint64_t i = 0; i < g->node_count; i++) {
                float a_abs = fabsf(g->nodes[i].a);
                if (a_abs > 0.001f) {
                    sum_activation += a_abs;
                    active_count++;
                }
            }
            
            for (uint64_t i = 0; i < g->edge_count; i++) {
                sum_edge_strength += fabsf(g->edges[i].w);
            }
            
            g->avg_activation = (active_count > 0) ? (sum_activation / active_count) : 0.1f;
            if (g->avg_activation < 0.01f) g->avg_activation = 0.1f;  /* Minimum */
            
            g->avg_edge_strength = (g->edge_count > 0) ? (sum_edge_strength / g->edge_count) : 0.1f;
            if (g->avg_edge_strength < 0.01f) g->avg_edge_strength = 0.1f;  /* Minimum */
            
            /* Estimate initial chaos from activation variance */
            float variance = 0.0f;
            uint64_t sample_count = (g->node_count < 100) ? g->node_count : 100;
            for (uint64_t i = 0; i < sample_count; i++) {
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
        
        /* Initialize event-driven propagation state */
        g->prop_queue_size = (g->node_count > 0) ? g->node_count : 256;
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
        
        g->last_activation = calloc(g->node_count, sizeof(float));
        g->last_message = calloc(g->node_count, sizeof(float));
        g->output_propensity = calloc(g->node_count, sizeof(float));
        g->feedback_correlation = calloc(g->node_count, sizeof(float));
        g->prediction_accuracy = calloc(g->node_count, sizeof(float));
        if (!g->last_activation || !g->last_message || !g->output_propensity || 
            !g->feedback_correlation || !g->prediction_accuracy) {
            /* Cleanup on failure */
            if (g->last_activation) free(g->last_activation);
            if (g->last_message) free(g->last_message);
            if (g->output_propensity) free(g->output_propensity);
            if (g->feedback_correlation) free(g->feedback_correlation);
            if (g->prediction_accuracy) free(g->prediction_accuracy);
            melvin_close(g);
            return NULL;
        }
        /* Initialize from current state */
        for (uint64_t i = 0; i < g->node_count; i++) {
            g->last_activation[i] = g->nodes[i].a;
            /* Initialize output_propensity from node's output_propensity field */
            g->output_propensity[i] = g->nodes[i].output_propensity;
        }
        g->avg_chaos = 0.1f;  /* Initial estimate */
        g->avg_activation = 0.1f;  /* Jump start: small initial activation to bootstrap */
        g->avg_edge_strength = 0.1f;
        g->avg_output_activity = 0.0f;
        g->avg_feedback_correlation = 0.0f;
        g->avg_prediction_accuracy = 0.0f;
        
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
    
    /* Free dynamic propagation queue */
    if (g->prop_queue) free(g->prop_queue);
    if (g->prop_queued) free(g->prop_queued);
    
    free(g);
}

/* ========================================================================
 * SET SYSCALLS (writes pointer into blob)
 * ======================================================================== */

/* Thread-local Graph* for syscalls (needed for sys_copy_from_cold) */
static __thread Graph* current_graph = NULL;

Graph* melvin_get_current_graph(void) {
    return current_graph;
}

void melvin_set_syscalls(Graph *g, MelvinSyscalls *syscalls) {
    current_graph = g;  /* Set thread-local context */
    if (!g || !g->hdr || !syscalls) return;
    
    /* Write syscalls pointer into blob at known offset */
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
    
    if (g->hdr->syscalls_ptr_offset > 0 && 
        g->hdr->syscalls_ptr_offset < g->hdr->blob_size) {
        void **ptr_loc = (void **)(g->blob + g->hdr->syscalls_ptr_offset);
        return (MelvinSyscalls *)*ptr_loc;
    }
    
    return NULL;
}

/* ========================================================================
 * FEED BYTE (ONLY writes to .m, NO physics)
 * ======================================================================== */

/* Forward declaration */
static void ensure_node(Graph *g, uint32_t node_id);

static uint32_t find_edge(Graph *g, uint32_t src, uint32_t dst) {
    /* Ensure nodes exist - graph grows dynamically */
    ensure_node(g, src);
    ensure_node(g, dst);
    uint32_t eid = g->nodes[src].first_out;
    uint32_t max_iterations = (uint32_t)(g->edge_count + 1);  /* Safety: prevent infinite loops */
    uint32_t iterations = 0;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iterations < max_iterations) {
        if (g->edges[eid].dst == dst) return eid;
        eid = g->edges[eid].next_out;
        iterations++;
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
        /* Fallback: unmap and remap (works on all platforms) */
        munmap(g->map_base, g->map_size);
        new_map = mmap(NULL, (size_t)new_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, g->fd, 0);
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
    
    /* Initialize new nodes */
    for (uint64_t i = old_node_count; i < new_node_count; i++) {
        memset(&g->nodes[i], 0, sizeof(Node));
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
    
    return 0;
}

/* Ensure node exists, grow if needed */
static void ensure_node(Graph *g, uint32_t node_id) {
    if (!g || node_id < g->node_count) return;
    
    /* Grow to at least node_id + 1, with some headroom */
    uint64_t new_count = (uint64_t)node_id + 1;
    if (new_count < g->node_count * 2) {
        new_count = g->node_count * 2;  /* Double for efficiency */
    }
    grow_nodes(g, new_count);
}

static uint32_t create_edge(Graph *g, uint32_t src, uint32_t dst, float w) {
    /* Simple edge creation - just structure, no physics */
    /* Hot region only - edges stay in hot space */
    /* NO LIMITS - graph controls its own growth */
    
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
            new_map = mmap(NULL, (size_t)new_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, g->fd, 0);
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

/* ========================================================================
 * UEL PHYSICS PARAMETERS (must be before event-driven code)
 * ======================================================================== */

/* UEL physics parameters - ALL RELATIVE (no hardcoded thresholds) */
static const struct {
    float eta_a_base;          /* Base learning rate (scaled by graph state) */
    float eta_w_base;          /* Base weight learning rate (scaled by graph state) */
    float lambda;              /* Global field coupling (relative to local messages) */
    float decay_a_base;        /* Base activation decay (scaled by graph state) */
    float decay_w_base;        /* Base weight decay (scaled by graph state) */
    float change_threshold_ratio;  /* Change threshold as ratio of avg_activation */
    float running_avg_alpha;   /* Exponential moving average decay (adaptive) */
    float weight_clamp_ratio;  /* Weight clamp as ratio of avg_edge_strength */
    float active_threshold_ratio; /* Active node threshold as ratio of avg_activation */
    
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
    .decay_a_base = 0.05f,     /* Base - will be scaled by graph state */
    .decay_w_base = 0.001f,    /* Base - will be scaled by graph state */
    .change_threshold_ratio = 0.1f,  /* 10% of avg_activation */
    .running_avg_alpha = 0.99f, /* Slow adaptation */
    .weight_clamp_ratio = 10.0f, /* 10x avg_edge_strength */
    .active_threshold_ratio = 0.1f,  /* 10% of avg_activation */
    
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
        for (uint64_t i = 0; i < g->node_count && i < 256; i++) {
            if (g->output_propensity[i] > 0.1f && fabsf(g->nodes[i].a) > g->avg_activation * 0.5f) {
                /* This node was an active output - reward it for creating feedback */
                g->feedback_correlation[i] = uel_params.running_avg_alpha * g->feedback_correlation[i] + 
                                            (1.0f - uel_params.running_avg_alpha) * feedback_strength;
            }
        }
    }
    
    /* Inject energy - this is the event that triggers propagation */
    g->nodes[port_node_id].a += energy;
    g->nodes[data_id].a += energy;
    
    /* Ensure edge exists (structure only) */
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
    uint32_t max_iterations = (uint32_t)(g->edge_count + 1);  /* Safety: prevent infinite loops */
    uint32_t iterations = 0;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iterations < max_iterations) {
        if (g->edges[eid].dst == dst) return eid;
        eid = g->edges[eid].next_out;
        iterations++;
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
    uint32_t max_iter_i = (uint32_t)(g->edge_count + 1);
    uint32_t iter_i = 0;
    
    while (ei != UINT32_MAX && ei < g->edge_count && iter_i < max_iter_i) {
        uint32_t neighbor = g->edges[ei].src;
        uint32_t ej = g->nodes[j].first_in;
        uint32_t max_iter_j = (uint32_t)(g->edge_count + 1);
        uint32_t iter_j = 0;
        
        while (ej != UINT32_MAX && ej < g->edge_count && iter_j < max_iter_j) {
            if (g->edges[ej].src == neighbor) return 0.3f;
            ej = g->edges[ej].next_in;
            iter_j++;
        }
        ei = g->edges[ei].next_in;
        iter_i++;
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

/* Compute message from neighbors for a single node */
static float compute_message(Graph *g, uint32_t node_id) {
    float msg = 0.0f;
    uint32_t eid = g->nodes[node_id].first_in;
    uint32_t max_iter = (uint32_t)(g->edge_count + 1);
    uint32_t iter = 0;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
        msg += g->edges[eid].w * g->nodes[g->edges[eid].src].a;
        eid = g->edges[eid].next_in;
        iter++;
    }
    return msg;
}

/* Compute global field contribution for a single node (simplified) */
static float compute_phi_contribution(Graph *g, uint32_t node_id, float *mass) {
    float phi = 0.0f;
    float mass_i = mass[node_id];
    if (mass_i < 0.001f) return 0.0f;
    
    /* Sample neighbors for efficiency (or use cached values) */
    uint32_t eid = g->nodes[node_id].first_in;
    uint32_t count = 0;
    uint32_t max_iter = (uint32_t)(g->edge_count + 1);
    uint32_t iter = 0;
    
    /* RELATIVE: sample count based on node degree (more neighbors = sample more) */
    /* Scale with degree: high-degree nodes need more samples for accurate phi */
    uint32_t degree = g->nodes[node_id].in_degree;
    uint32_t max_samples = (degree < 5) ? degree : (5 + (degree / 5));  /* Progressive scaling */
    /* Cap relative to graph size to prevent excessive sampling */
    uint32_t graph_based_max = (g->node_count > 100) ? (uint32_t)(g->node_count / 100) : 50;
    if (max_samples > graph_based_max) max_samples = graph_based_max;
    
    while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter && count < max_samples) {
        uint32_t src = g->edges[eid].src;
        /* RELATIVE: active threshold based on avg_activation */
        float active_threshold = g->avg_activation * uel_params.active_threshold_ratio;
        if (active_threshold < 0.001f) active_threshold = 0.001f;  /* Minimum */
        
        if (src < g->node_count && mass[src] > active_threshold) {
            /* Simplified kernel: direct connection */
            phi += mass[src] * 0.5f;
            count++;
        }
        eid = g->edges[eid].next_in;
        iter++;
    }
    return phi;
}

/* Update a single node and propagate if changed significantly */
static void update_node_and_propagate(Graph *g, uint32_t node_id, float *mass) {
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
    g->avg_chaos = uel_params.running_avg_alpha * g->avg_chaos + 
                   (1.0f - uel_params.running_avg_alpha) * chaos_i;
    /* Clamp activation for running average */
    float a_abs = fabsf(a_i);
    if (a_abs > 10.0f) a_abs = 10.0f;
    g->avg_activation = uel_params.running_avg_alpha * g->avg_activation + 
                       (1.0f - uel_params.running_avg_alpha) * a_abs;
    
    /* Relative chaos: how chaotic relative to graph average */
    float relative_chaos = (g->avg_chaos > 0.001f) ? (chaos_i / g->avg_chaos) : chaos_i;
    
    /* Gradient descent: move a_i toward field_input to reduce chaos */
    /* Learning rate adapts to graph state (faster when chaotic) */
    /* RELATIVE: eta_a scales with relative_chaos and avg_activation */
    float adaptive_eta = uel_params.eta_a_base * (1.0f + relative_chaos * 0.5f) * 
                         (1.0f / (1.0f + g->avg_activation));
    float da_i = -adaptive_eta * (a_i - field_input);
    
    /* RESTLESSNESS: Drive to discharge high activation */
    float restlessness_pressure = 0.0f;
    float restlessness_threshold = g->avg_activation * uel_params.restlessness_threshold_ratio;
    if (restlessness_threshold < 0.001f) restlessness_threshold = 0.001f;
    if (a_i > restlessness_threshold) {
        restlessness_pressure = -uel_params.restlessness_strength * (a_i - restlessness_threshold);
    }

    /* ENERGY QUALITY: Reward coherent high energy */
    float energy_quality = (fabsf(a_i) > 0.001f) ? (fabsf(a_i) / (1.0f + chaos_i)) : 0.0f;
    float quality_reward = uel_params.quality_strength * energy_quality;

    /* PREDICTION: Reward accurate predictions (simplified for now) */
    float prediction_reward = uel_params.prediction_strength * g->prediction_accuracy[node_id];
    
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
    
    /* Update with decay (activation cost term) */
    /* RELATIVE: decay scales with avg_activation (more active = more decay cost) */
    float relative_decay_a = uel_params.decay_a_base * (1.0f + g->avg_activation);
    float new_a = a_i + da_i - relative_decay_a * a_i;
    g->nodes[node_id].a = tanhf(new_a);
    
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
            if (g->output_propensity[node_id] > 0.1f && g->feedback_correlation[node_id] > 0.0f) {
                float exploration_boost = uel_params.exploration_strength * 
                                         g->output_propensity[node_id] * 
                                         g->feedback_correlation[node_id] * 
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
    float activation_change = fabsf(new_a - g->last_activation[node_id]);
    float message_change = fabsf(msg_i - g->last_message[node_id]);
    
    /* Relative threshold: change must be significant compared to graph's typical activation */
    float relative_change_threshold = g->avg_activation * uel_params.change_threshold_ratio;
    if (relative_change_threshold < 0.001f) relative_change_threshold = 0.001f;  /* Minimum */
    
    if (activation_change > relative_change_threshold || 
        message_change > relative_change_threshold) {
        /* Significant change - propagate to neighbors */
        g->last_activation[node_id] = new_a;
        g->last_message[node_id] = msg_i;
        
            /* Add outgoing neighbors to queue */
            eid = g->nodes[node_id].first_out;
            iter = 0;
            while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
                uint32_t dst = g->edges[eid].dst;
                ensure_node(g, dst);  /* Ensure node exists - graph grows dynamically */
                if (dst < g->node_count) {
                    prop_queue_add(g, dst);
                }
                eid = g->edges[eid].next_out;
                iter++;
            }
            
            /* Add incoming neighbors to queue (they need to update their messages) */
            eid = g->nodes[node_id].first_in;
            iter = 0;
            while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
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
            uint64_t read_length = 100;  /* Read 100 bytes */
            if (cold_offset + read_length > g->cold_data_size) {
                read_length = g->cold_data_size - cold_offset;
            }
            
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
            
            if (node_id < g->node_count && 
                g->last_activation[node_id] > recent_threshold) {
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
static void uel_main(Graph *g) {
    /* PURE EVENT-DRIVEN: Only process nodes in the propagation queue */
    /* No global ticks - only process what's queued */
    if (!g || !g->hdr || g->node_count == 0) return;
    
    /* Allocate mass buffer for phi computation */
    float *mass = calloc(g->node_count, sizeof(float));
    if (!mass) return;
    
    /* Compute mass for all nodes (needed for phi computation) */
    for (uint64_t i = 0; i < g->node_count; i++) {
        float a_abs = fabsf(g->nodes[i].a);
        float degree = (float)(g->nodes[i].in_degree + g->nodes[i].out_degree);
        float degree_scale = g->avg_activation * 0.1f;
        if (degree_scale < 0.01f) degree_scale = 0.01f;
        mass[i] = a_abs + degree_scale * degree;
    }
    
    /* Curiosity: reactivate nodes when bored (creates natural exploration) */
    /* This adds nodes to the queue, which then get processed */
    curiosity_reactivate(g);
    
    /* Process all queued nodes (event-driven, not tick-based) */
    /* CONTINUOUS: Process until queue is truly empty, no artificial limits */
    /* Graph controls its own activity through UEL physics - high chaos = more processing */
    uint32_t processed = 0;
    uint32_t consecutive_empty = 0;
    /* RELATIVE: Empty check threshold scales with graph size */
    /* Larger graphs may need more checks as propagation takes longer */
    uint32_t base_empty_checks = 10;
    float graph_size_factor = (g->node_count > 0) ? (1.0f + (float)g->node_count / 1000.0f) : 1.0f;
    if (graph_size_factor > 5.0f) graph_size_factor = 5.0f;  /* Cap at 5x */
    uint32_t max_consecutive_empty = (uint32_t)(base_empty_checks * graph_size_factor);
    if (max_consecutive_empty < 5) max_consecutive_empty = 5;  /* Minimum */
    
    /* Process queue until truly empty (continuous, not tick-limited) */
    while (consecutive_empty < max_consecutive_empty) {
        uint32_t node_id = prop_queue_get(g);
        if (node_id == UINT32_MAX) {
            /* Queue empty - check if curiosity added anything */
            curiosity_reactivate(g);
            node_id = prop_queue_get(g);
            if (node_id == UINT32_MAX) {
                consecutive_empty++;
                if (consecutive_empty >= max_consecutive_empty) break;  /* Truly empty, done */
                continue;
            }
        }
        
        /* Reset empty counter - we found work */
        consecutive_empty = 0;
        
        /* Process node (event-driven) */
        update_node_and_propagate(g, node_id, mass);
        processed++;
        
        /* SELF-REGULATION: Graph controls its own activity through UEL */
        /* High chaos = more processing needed, but also more energy cost */
        /* Graph learns to balance through UEL physics */
        
        /* Safety: prevent truly infinite loops (only if queue keeps refilling with same nodes) */
        /* RELATIVE: Safety limit scales with graph size - larger graphs naturally process more */
        /* This is a last resort - graph should learn to stabilize through UEL */
        uint64_t safety_limit = (g->node_count > 0) ? (g->node_count * 1000) : 1000000;
        if (safety_limit < 100000) safety_limit = 100000;  /* Minimum for small graphs */
        if (processed > (uint32_t)safety_limit) {
            /* Graph is in unstable state - break to prevent infinite loop */
            /* Graph should learn to stabilize through UEL (high chaos → learning) */
            break;
        }
        
        /* SELF-REGULATION: If chaos is very low and we've processed a lot, graph is stable */
        /* Graph naturally reduces activity when stable (through UEL decay) */
        if (g->avg_chaos < 0.01f && processed > 1000) {
            /* Graph is very stable - reduce processing (graph self-regulates) */
            /* This is natural - low chaos means graph has learned patterns */
            break;
        }
    }
    
    /* SELF-REGULATION: Update output activity tracking for feedback learning */
    /* Graph learns which outputs create feedback through this tracking */
    if (g->avg_output_activity > 0.0f) {
        /* Decay output activity over time (graph forgets old outputs) */
        g->avg_output_activity *= 0.99f;  /* Slow decay */
    }
    
    free(mass);
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
    
    /* mmap */
    void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
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
    
    /* Initialize data nodes (0-255) - just structure, no physics */
    Node *nodes = (Node *)((char *)map + hdr->nodes_offset);
    for (int i = 0; i < 256 && i < (int)hot_nodes; i++) {
        nodes[i].byte = (uint8_t)i;
        nodes[i].a = 0.0f;
        nodes[i].first_in = UINT32_MAX;
        nodes[i].first_out = UINT32_MAX;
        nodes[i].in_degree = 0;
        nodes[i].out_degree = 0;
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
    uel_main(g);
    
    /* GRAPH-DRIVEN BLOB EXECUTION: Execute blob code when output nodes activate */
    /* Graph decides when to run its own code through activation patterns */
    /* Check if any output nodes are highly activated - if so, blob code might want to run */
    bool should_execute_blob = false;
    float max_output_activation = 0.0f;
    
    /* Scan output ports (100-199) for high activation */
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
    
    /* Also check tool gateway outputs (they produce patterns) */
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
    
    /* Execute blob code if graph decided to (through activation) */
    /* Graph learns when blob execution is useful through UEL feedback */
    if (should_execute_blob && g->hdr->main_entry_offset >= 0 && 
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

