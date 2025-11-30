# Pure Substrate Philosophy

## Core Principle

**`melvin.c` is pure substrate** - just the graph physics engine (UEL). 

**Tools (LLM, STT, TTS, Vision) are external pattern generators** - discovered via syscalls, not hardcoded.

## What This Means

### ✅ In melvin.c (Pure Substrate):
- Generic tool gateway ranges (300-699) - abstract, no specific tools
- Generic input/output propensities - tool-like behavior hints
- Generic edge patterns - very weak, graph discovers which tools
- UEL physics - pure graph dynamics
- No tool names, no tool-specific logic

### ❌ NOT in melvin.c:
- Specific tool names (STT, LLM, TTS, Vision)
- Tool-specific edge patterns
- Tool-specific comments
- Hardcoded tool routing

### ✅ In host_syscalls.c / melvin_tools.c:
- Tool implementations (Ollama, Whisper, Piper, ONNX)
- Tool discovery via syscalls
- Tool outputs → graph patterns

## Why This Matters

1. **Tools are optional** - Graph can learn without them (just slower)
2. **Tools are discovered** - Graph learns which tools exist via syscalls
3. **Tools are replaceable** - Swap Ollama for GPT, Whisper for Vosk, etc.
4. **Pure emergence** - Graph builds understanding from patterns, not hardcoded knowledge

## Current Status

**melvin.c is being cleaned** - removing all tool-specific references to make it pure substrate.

Tools remain in `melvin_tools.c` and are discovered via the syscall interface.

