#!/usr/bin/env python3
"""
Simple LLM → Brain Demo
Shows real Llama 3 output being injected into brain.m
"""

import subprocess
import os

def query_llama(prompt):
    """Get knowledge from Llama 3"""
    print(f"🤖 Asking Llama 3: '{prompt}'\n")
    
    cmd = ["ollama", "run", "llama3.2:1b", prompt]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except:
        return "ERROR"

def inject_knowledge(brain_file, knowledge):
    """Inject knowledge into brain using Melvin API"""
    print(f"📝 Injecting: '{knowledge}'\n")
    
    # Simple program that feeds text into brain
    program = """
#include "src/melvin.h"
#include <stdio.h>
#include <string.h>

int main() {
    Graph *brain = melvin_open("llm_brain.m", 10000, 50000, 131072);
    if (!brain) {
        printf("Creating brain...\\n");
        melvin_create_v2("llm_brain.m", 10000, 50000, 131072, 0);
        brain = melvin_open("llm_brain.m", 10000, 50000, 131072);
    }
    if (!brain) return 1;
    
    FILE *f = fopen("/tmp/llm_knowledge.txt", "r");
    if (!f) return 1;
    
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        for (char *p = line; *p; p++) {
            melvin_feed_byte(brain, 0, *p, 1.0f);
        }
        
        for (int i = 0; i < 5; i++) {
            melvin_call_entry(brain);
        }
    }
    fclose(f);
    
    melvin_close(brain);
    printf("Done\\n");
    return 0;
}
"""
    
    # Write knowledge to temp file
    with open('/tmp/llm_knowledge.txt', 'w') as f:
        f.write(knowledge)
    
    # Write program
    with open('/tmp/inject_llm.c', 'w') as f:
        f.write(program)
    
    # Compile
    subprocess.run(
        "cd /home/melvin/teachable_system && gcc /tmp/inject_llm.c melvin.o -O2 -I. -lm -lpthread -o /tmp/inject_llm 2>&1",
        shell=True, capture_output=True
    )
    
    # Run
    result = subprocess.run("/tmp/inject_llm", shell=True, capture_output=True, text=True)
    print(result.stdout)

def main():
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  LLAMA 3 → MELVIN BRAIN: LIVE DEMO                    ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    
    brain_file = "llm_brain.m"
    
    # Show initial state
    if os.path.exists(brain_file):
        size = os.path.getsize(brain_file)
        print(f"📊 Brain before: {size/1024/1024:.2f} MB\n")
    else:
        print("📊 Brain: Creating new...\n")
    
    # Query Llama 3
    print("═══ STEP 1: Query Llama 3 ═══")
    prompt = "Give me 3 simple robot rules in format: when X then Y. Each rule one line, under 10 words."
    
    llm_response = query_llama(prompt)
    
    print("Llama 3 says:")
    print("─" * 50)
    print(llm_response)
    print("─" * 50)
    print()
    
    # Inject into brain
    print("═══ STEP 2: Inject into brain.m ═══")
    inject_knowledge(brain_file, llm_response)
    
    # Show result
    if os.path.exists(brain_file):
        size = os.path.getsize(brain_file)
        print(f"\n📊 Brain after: {size/1024/1024:.2f} MB")
    
    print()
    print("✅ LLM knowledge now part of brain.m!")
    print(f"   File: /home/melvin/teachable_system/{brain_file}")
    print()

if __name__ == "__main__":
    main()

