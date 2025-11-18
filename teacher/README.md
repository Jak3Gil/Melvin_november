# Melvin Kindergarten Teacher

A dynamic teaching system that uses Ollama (local LLM) to teach Melvin simple patterns through a curriculum-based learning loop.

## Overview

The Kindergarten Teacher is a Python-based harness that:

- **Generates curriculum**: Uses Ollama to create simple teaching tasks (numbers, letters, patterns)
- **Teaches Melvin**: Feeds tasks to Melvin's learning system
- **Evaluates progress**: Uses Ollama to judge how well Melvin learned each pattern
- **Logs results**: Saves all interactions to `teacher_log.jsonl` for analysis

## Prerequisites

1. **Ollama installed and running**:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3
   ollama serve  # Run in background or separate terminal
   ```

2. **Python 3** with `requests`:
   ```bash
   pip install requests
   ```

3. **Melvin compiled**:
   ```bash
   make learn  # Builds melvin_learn_cli
   ```

## Usage

### Basic Usage

```bash
cd teacher
python3 kindergarten_teacher.py
```

### With Options

```bash
python3 kindergarten_teacher.py --rounds 20 --tasks-per-round 5
```

### Custom Melvin Binary

```bash
python3 kindergarten_teacher.py --melvin-binary ../melvin_learn_cli
```

## How It Works

1. **Task Generation**: Ollama generates simple tasks like:
   - `"1 2 3 4 5"` (counting)
   - `"ababab"` (repeating pattern)
   - `"a b c d"` (alphabet sequence)

2. **Melvin Learning**: Each task is fed to `melvin_learn_cli` which:
   - Creates DATA nodes from the input string
   - Generates patterns (bigrams, trigrams)
   - Runs 10 self-consistency episodes
   - Outputs JSON with patterns, compression ratio, error

3. **Evaluation**: Ollama judges Melvin's output:
   - Compares discovered patterns to expected pattern hint
   - Returns score (0.0 to 1.0) and feedback

4. **Logging**: All results saved to `teacher_log.jsonl`:
   ```json
   {
     "round": 1,
     "task": {
       "description": "counting numbers",
       "input_str": "1 2 3 4 5",
       "expected_pattern_hint": "sequential numbers"
     },
     "melvin_result": {
       "input_str": "1 2 3 4 5",
       "compression_ratio": 0.600,
       "reconstruction_error": 0.000,
       "patterns": [...]
     },
     "judge": {
       "score": 0.85,
       "feedback": "Melvin successfully identified the sequential pattern"
     }
   }
   ```

## Files

- `ollama_client.py` - Simple wrapper for Ollama API
- `kindergarten_teacher.py` - Main teaching loop
- `teacher_log.jsonl` - Generated log file (created automatically)

## Troubleshooting

### Ollama Connection Error

If you see "Ollama API error", check:
- Ollama is running: `curl http://localhost:11434/api/tags`
- Model is available: `ollama list`
- Default model name matches: Edit `OLLAMA_MODEL` in `ollama_client.py`

### Melvin Binary Not Found

- Build it: `make learn`
- Check path: `ls -la melvin_learn_cli`
- Use `--melvin-binary` flag to specify custom path

### JSON Parse Errors

If Ollama returns malformed JSON:
- The system falls back to predefined tasks
- Check Ollama output in terminal
- Try a different model or adjust prompts in `kindergarten_teacher.py`

## Next Steps

The teacher system is designed to be extended:

- **Adaptive curriculum**: Adjust task difficulty based on judge scores
- **Pattern analysis**: Analyze which patterns Melvin discovers most often
- **Visualization**: Plot learning curves from `teacher_log.jsonl`
- **Multi-model**: Compare different Ollama models as teachers

## Notes

- The substrate (C code) is **never modified** - all teaching logic is in Python
- Melvin remains a general pattern engine - Ollama handles semantics
- This is a "kindergarten" level - very simple patterns to establish foundations

