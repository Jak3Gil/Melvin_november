#ifndef MC_SCAFFOLD_H
#define MC_SCAFFOLD_H

#include "../melvin.h"

// Scanned rule from scaffold comment
typedef struct {
    char name[128];
    char rule_type[64];  // SAFETY_RULE, BEHAVIOR_RULE, etc.
    char context[512];
    char effect[512];
    char origin_file[256];
} ScannedRule;

// Channel IDs for semantic connections
#define CH_VISION   1
#define CH_SENSOR   2
#define CH_MOTOR    3
#define CH_REWARD   4
#define CH_META     5

// Origin types stored in node flags
#define ORIGIN_SCAFFOLD  (1 << 16)

// Function declarations
void mc_scaffold_emit_rule(Brain *g, const ScannedRule *r);
void mc_scaffold_process_file(Brain *g, const char *file_path);
void mc_scaffold_cleanup(void);
int mc_scaffold_should_scan(Brain *g);

#endif // MC_SCAFFOLD_H

