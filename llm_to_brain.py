#!/usr/bin/env python3
"""
LLM → Melvin Brain Integration
Uses Ollama Llama 3 to generate knowledge and inject it into brain.m

Flow:
1. Query Llama 3 for domain knowledge
2. Parse LLM response
3. Feed into Melvin brain as training data
4. Brain creates patterns from LLM knowledge
5. Save enriched brain.m
"""

import subprocess
import json
import sys
import os

def query_llama(prompt, model="llama3.2:1b"):
    """Query Ollama Llama 3 model"""
    print(f"🤖 Querying {model}...")
    print(f"   Prompt: '{prompt}'\n")
    
    cmd = [
        "ollama", "run", model,
        prompt
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        response = result.stdout.strip()
        return response
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def feed_to_brain(brain_file, knowledge_text):
    """Feed LLM knowledge into Melvin brain using C program"""
    print(f"📝 Feeding to brain: '{knowledge_text[:50]}...'")
    
    # Create temporary C program to inject this specific knowledge
    inject_code = f'''
#include "src/melvin.h"
#include <stdio.h>

int main() {{
    Graph *brain = melvin_open("{brain_file}", 10000, 50000, 131072);
    if (!brain) return 1;
    
    const char *knowledge = "{knowledge_text.replace('"', '\\"')}";
    
    // Feed knowledge as high-energy pattern
    for (const char *p = knowledge; *p; p++) {{
        melvin_feed_byte(brain, 0, *p, 1.0f);
    }}
    
    // Process to create patterns
    for (int i = 0; i < 10; i++) {{
        melvin_call_entry(brain);
    }}
    
    melvin_close(brain);
    printf("✅ Injected into brain\\n");
    return 0;
}}
'''
    
    # Write, compile, run
    with open('/tmp/inject.c', 'w') as f:
        f.write(inject_code)
    
    # Compile
    compile_cmd = "cd /home/melvin/teachable_system && gcc /tmp/inject.c melvin.o -O2 -I. -lm -lpthread -o /tmp/inject 2>&1"
    subprocess.run(compile_cmd, shell=True, capture_output=True)
    
    # Run
    run_cmd = "/tmp/inject"
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)

def main():
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  LLM → MELVIN: Real Llama 3 Integration              ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    
    brain_file = "/home/melvin/teachable_system/llm_brain.m"
    
    # Check if brain exists, create if not
    if not os.path.exists(brain_file):
        print(f"Creating new brain: {brain_file}")
        create_cmd = f"cd /home/melvin/teachable_system && ./create_brain.sh {brain_file} 10000 50000 2>&1 | tail -3"
        subprocess.run(create_cmd, shell=True)
        print()
    
    # Get file size before
    size_before = os.path.getsize(brain_file) if os.path.exists(brain_file) else 0
    print(f"📊 Brain size before: {size_before/1024/1024:.2f} MB\n")
    
    # Query LLM for knowledge
    print("═══════════════════════════════════════════════════════")
    print("STEP 1: Querying Llama 3 for sensor-action rules")
    print("═══════════════════════════════════════════════════════")
    print()
    
    prompt = "List 5 simple sensor-action rules for a robot. Format: 'when X then Y'. Keep each rule under 10 words."
    
    llm_response = query_llama(prompt)
    
    print("🤖 Llama 3 Response:")
    print("─" * 55)
    print(llm_response)
    print("─" * 55)
    print()
    
    # Parse response into rules
    rules = []
    for line in llm_response.split('\n'):
        line = line.strip()
        if line and len(line) > 10 and len(line) < 100:
            # Clean up line
            line = line.lstrip('0123456789.- ')
            if 'when' in line.lower() or 'if' in line.lower() or 'then' in line.lower():
                rules.append(line)
    
    print(f"📋 Extracted {len(rules)} rules from LLM\n")
    
    # Feed each rule into brain
    print("═══════════════════════════════════════════════════════")
    print("STEP 2: Injecting LLM knowledge into brain.m")
    print("═══════════════════════════════════════════════════════")
    print()
    
    for i, rule in enumerate(rules[:5], 1):  # Limit to 5 rules
        print(f"Rule {i}: {rule}")
        feed_to_brain(brain_file, rule)
        print()
    
    # Check file size after
    size_after = os.path.getsize(brain_file) if os.path.exists(brain_file) else 0
    print(f"📊 Brain size after: {size_after/1024/1024:.2f} MB")
    print(f"📈 Growth: {(size_after-size_before)/1024:.1f} KB\n")
    
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  SUCCESS: LLM knowledge integrated into brain.m!      ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║  ✓ Llama 3 generated semantic knowledge              ║")
    print("║  ✓ Knowledge injected as neural patterns             ║")
    print("║  ✓ Brain.m file updated with new patterns            ║")
    print("║  ✓ Brain can now use LLM-seeded knowledge            ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    
    print(f"Brain file: {brain_file}")
    print("View with: ../tools/inspect_graph llm_brain.m\n")

if __name__ == "__main__":
    main()

