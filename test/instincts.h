#ifndef INSTINCTS_H
#define INSTINCTS_H

// Forward declarations (types defined in melvin.c)
struct MelvinFile;

// Main injection function - call this after file creation to inject initial patterns
// Injects: param nodes, channel patterns, code patterns, reward patterns, body patterns
// All patterns are regular nodes + edges + bytes - no special types
void melvin_inject_instincts(struct MelvinFile *file);

// Individual injection functions (internal, but exposed for testing if needed)
// These are static in instincts.c - not meant for external use
// Use melvin_inject_instincts() instead

#endif // INSTINCTS_H

