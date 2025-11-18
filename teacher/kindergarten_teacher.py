"""
Melvin Kindergarten Teacher - Dynamic teaching loop using Ollama.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
from ollama_client import ollama_chat, OLLAMA_MODEL

SYSTEM_TEACHER = (
    "You are a kindergarten teacher for a tiny graph-based learner called Melvin.\n"
    "You must produce very short, simple string patterns that teach basics like numbers,"
    " letters, and simple math patterns. The output must be a JSON list of tasks, where"
    " each task has fields: description, input_str, expected_pattern_hint.\n"
    "Keep strings ASCII-only and under 40 characters.\n"
    "Example format: [{\"description\": \"counting numbers\", \"input_str\": \"1 2 3 4 5\", \"expected_pattern_hint\": \"sequential numbers\"}]"
)

SYSTEM_JUDGE = (
    "You are evaluating a tiny pattern-learning system called Melvin.\n"
    "You see a teaching task and Melvin's internal report (patterns + metrics).\n"
    "Your job is to estimate how well Melvin captured the intended pattern (0 to 1),"
    " and give short feedback.\n"
    "Respond as strict JSON: {\"score\": float, \"feedback\": \"...\"}."
)


@dataclass
class Task:
    description: str      # natural language description (for logging / Ollama)
    input_str: str        # the byte string we feed to Melvin
    expected_pattern_hint: str  # human/LLM-level hint, not used by Melvin directly


def generate_kindergarten_tasks(num_tasks: int = 5) -> List[Task]:
    """Use Ollama to generate simple teaching tasks."""
    user_prompt = (
        f"Generate {num_tasks} very simple teaching tasks for pattern learning.\n"
        "Each task should have a description, a short input string (ASCII, <40 chars),"
        " and a hint about the expected pattern.\n"
        "Return ONLY valid JSON array, no other text."
    )
    
    content = ollama_chat(SYSTEM_TEACHER, user_prompt)
    
    if not content:
        print("Ollama returned empty response, using fallback tasks")
        return _fallback_tasks(num_tasks)
    
    # Try to extract JSON from the response
    tasks_data = []
    try:
        # Try to find JSON array in the response
        start = content.find('[')
        end = content.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            tasks_data = json.loads(json_str)
        else:
            tasks_data = json.loads(content)
    except json.JSONDecodeError:
        print("Failed to parse JSON from Ollama. Raw content:")
        print(content)
        print("Using fallback tasks")
        return _fallback_tasks(num_tasks)
    
    tasks: List[Task] = []
    for item in tasks_data:
        try:
            t = Task(
                description=item.get("description", ""),
                input_str=item.get("input_str", ""),
                expected_pattern_hint=item.get("expected_pattern_hint", "")
            )
            if t.input_str:  # Only add if we have actual input
                tasks.append(t)
        except (KeyError, TypeError) as e:
            print(f"Skipping invalid task item: {e}")
            continue
    
    if not tasks:
        print("No valid tasks generated, using fallback")
        return _fallback_tasks(num_tasks)
    
    return tasks


def _fallback_tasks(num_tasks: int = 5) -> List[Task]:
    """Fallback tasks if Ollama fails."""
    all_tasks = [
        Task("counting numbers", "1 2 3 4 5", "sequential numbers"),
        Task("alphabet pattern", "a b c d", "sequential letters"),
        Task("repeated pattern", "ababab", "alternating ab"),
        Task("even numbers", "2 4 6 8", "even number sequence"),
        Task("simple math", "1+1=2", "addition pattern"),
    ]
    # Return only the requested number, cycling if needed
    return all_tasks[:num_tasks] if num_tasks <= len(all_tasks) else all_tasks * ((num_tasks // len(all_tasks)) + 1)


# Global persistent Melvin process
_persistent_melvin_proc = None
_persistent_melvin_graph_file = None

def start_persistent_melvin(melvin_binary: str, graph_file: str = None):
    """Start Melvin as a persistent process (loads graph once)."""
    global _persistent_melvin_proc, _persistent_melvin_graph_file
    
    if _persistent_melvin_proc is not None:
        # Already running, check if it's the same graph
        if _persistent_melvin_graph_file == graph_file:
            return  # Same graph, reuse process
        else:
            # Different graph, restart
            _persistent_melvin_proc.terminate()
            _persistent_melvin_proc.wait()
            _persistent_melvin_proc = None
    
    cmd = [melvin_binary]
    if graph_file:
        cmd.extend(["--load", graph_file, "--save", graph_file])
    
    _persistent_melvin_proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    _persistent_melvin_graph_file = graph_file
    
    # Wait a moment for startup (graph loading happens here)
    import time
    time.sleep(0.1)

def stop_persistent_melvin():
    """Stop the persistent Melvin process."""
    global _persistent_melvin_proc
    if _persistent_melvin_proc:
        _persistent_melvin_proc.stdin.close()
        _persistent_melvin_proc.terminate()
        _persistent_melvin_proc.wait()
        _persistent_melvin_proc = None

def run_melvin_on_string(input_str: str, melvin_binary: str = "./melvin_learn_cli",
                         graph_file: str = None, use_persistent: bool = True) -> dict:
    """Run Melvin on a string and parse JSON output.
    
    If use_persistent=True, uses a persistent process (graph loaded once).
    If use_persistent=False, spawns a new process per call (slow, loads graph each time).
    """
    global _persistent_melvin_proc
    
    if use_persistent:
        # Use persistent process (fast - graph loaded once)
        if _persistent_melvin_proc is None:
            start_persistent_melvin(melvin_binary, graph_file)
        
        try:
            # Send input
            _persistent_melvin_proc.stdin.write(input_str + "\n")
            _persistent_melvin_proc.stdin.flush()
            
            # Read output (JSON response)
            # Read until we get a complete JSON object (ends with }\n)
            output_lines = []
            brace_count = 0
            in_json = False
            
            while True:
                line = _persistent_melvin_proc.stdout.readline()
                if not line:
                    break
                
                output_lines.append(line.rstrip())
                
                # Track braces to know when JSON is complete
                for char in line:
                    if char == '{':
                        brace_count += 1
                        in_json = True
                    elif char == '}':
                        brace_count -= 1
                        if in_json and brace_count == 0:
                            # Complete JSON object
                            break
                
                if in_json and brace_count == 0:
                    break
            
            out = '\n'.join(output_lines)
            
            # Also read stderr (timing info) - non-blocking
            # Note: select might not work on all platforms, so we skip it
            # The stderr timing info is logged but doesn't block
            
            if not out:
                return {"error": "No output from Melvin"}
            
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                print(f"Failed to parse Melvin JSON. Raw output:\n{out}")
                return {"error": "JSON parse error", "raw_output": out}
        except Exception as e:
            return {"error": f"Exception: {e}"}
    else:
        # Old behavior: spawn new process each time (slow)
        try:
            cmd = [melvin_binary]
            if graph_file:
                cmd.extend(["--load", graph_file, "--save", graph_file])
            
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = proc.communicate(input_str + "\n", timeout=60)
            
            if err:
                print(f"Melvin stderr: {err}")
            
            if not out:
                return {"error": "No output from Melvin"}
            
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                print(f"Failed to parse Melvin JSON. Raw output:\n{out}")
                return {"error": "JSON parse error", "raw_output": out}
        except subprocess.TimeoutExpired:
            return {"error": "Melvin timeout"}
        except FileNotFoundError:
            return {"error": f"Melvin binary not found: {melvin_binary}"}
        except Exception as e:
            return {"error": f"Exception: {e}"}


def evaluate_melvin_output(task: Task, melvin_result: dict) -> dict:
    """Ask Ollama to evaluate Melvin's output."""
    user_prompt = (
        f"Task description: {task.description}\n"
        f"Expected pattern hint: {task.expected_pattern_hint}\n"
        f"Input string: {task.input_str}\n\n"
        f"Melvin report (JSON):\n{json.dumps(melvin_result, indent=2)}\n\n"
        "Rate how well Melvin captured the pattern (0.0 to 1.0) and give brief feedback."
    )
    
    content = ollama_chat(SYSTEM_JUDGE, user_prompt)
    
    if not content:
        return {"score": 0.0, "feedback": "Ollama evaluation failed"}
    
    try:
        # Try to extract JSON from response
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            result = json.loads(json_str)
            return result
        else:
            result = json.loads(content)
            return result
    except json.JSONDecodeError:
        print("Failed to parse judge JSON. Raw content:")
        print(content)
        return {"score": 0.0, "feedback": "JSON parse error"}


def kindergarten_loop(num_rounds: int = 10, tasks_per_round: int = 3, 
                     melvin_binary: str = "./melvin_learn_cli",
                     persistent_graph: bool = True,
                     graph_file: str = None):
    """Main kindergarten training loop."""
    print("=== Melvin Kindergarten Teacher ===\n")
    print(f"Running {num_rounds} rounds with {tasks_per_round} tasks per round")
    if persistent_graph:
        if graph_file:
            print(f"Using persistent graph: {graph_file}\n")
        else:
            print("Using persistent global graph\n")
    else:
        print("Using fresh graph per task\n")
    
    log_file = "teacher_log.jsonl"
    if graph_file is None:
        graph_file = "melvin_global_graph.bin" if persistent_graph else None
    
    # Start persistent Melvin process (loads graph once)
    if persistent_graph:
        print("Starting persistent Melvin process (this will load the graph once)...")
        start_persistent_melvin(melvin_binary, graph_file)
        print("Melvin process started. Graph loaded.\n")
    
    import time
    start_time = time.time()
    
    for round_idx in range(num_rounds):
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        total_tasks = (round_idx * tasks_per_round)
        progress_pct = (round_idx / num_rounds * 100) if num_rounds > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_idx + 1}/{num_rounds} | Progress: {progress_pct:.1f}% | Elapsed: {elapsed_str}")
        print(f"{'='*70}")
        tasks = generate_kindergarten_tasks(tasks_per_round)
        
        if not tasks:
            print("No tasks generated, aborting.")
            break
        
        for t_idx, task in enumerate(tasks):
            task_num = (round_idx * tasks_per_round) + t_idx + 1
            total_tasks = num_rounds * tasks_per_round
            task_progress = (task_num / total_tasks * 100) if total_tasks > 0 else 0
            bar_width = 40
            filled = int(bar_width * task_num / total_tasks) if total_tasks > 0 else 0
            bar = '█' * filled + '░' * (bar_width - filled)
            
            print(f"\n--- Task {task_num}/{total_tasks}: {task.description} ---")
            print(f"[{bar}] {task_progress:.1f}%")
            print(f"Input: {task.input_str}")
            print(f"Expected: {task.expected_pattern_hint}")
            
            melvin_result = run_melvin_on_string(task.input_str, melvin_binary, graph_file, 
                                                 use_persistent=persistent_graph)
            
            if "error" in melvin_result:
                print(f"Melvin error: {melvin_result['error']}")
                judge_result = {"score": 0.0, "feedback": "Melvin execution failed"}
            else:
                # Show Melvin's output
                print(f"📤 Melvin Output:")
                print(f"   Patterns created: {melvin_result.get('num_patterns', 0)}")
                print(f"   Explanation apps: {melvin_result.get('explanation_apps', 0)}")
                print(f"   Compression ratio: {melvin_result.get('compression_ratio', 0):.3f}")
                print(f"   Reconstruction error: {melvin_result.get('reconstruction_error', 0):.3f}")
                patterns = melvin_result.get('patterns', [])
                if patterns:
                    top_pattern = sorted(patterns, key=lambda p: p.get('binding_count', 0), reverse=True)[0]
                    print(f"   Top pattern: ID={top_pattern.get('id', '?')}, "
                          f"q={top_pattern.get('q', 0):.3f}, "
                          f"bindings={top_pattern.get('binding_count', 0)}")
                print()
                judge_result = evaluate_melvin_output(task, melvin_result)
            
            score = judge_result.get("score", 0.0)
            feedback = judge_result.get("feedback", "No feedback")
            
            print(f"Judge score: {score:.2f}")
            print(f"Judge feedback: {feedback}")
            
            # Log to JSONL
            log_entry = {
                "round": round_idx + 1,
                "task": {
                    "description": task.description,
                    "input_str": task.input_str,
                    "expected_pattern_hint": task.expected_pattern_hint,
                },
                "melvin_result": melvin_result,
                "judge": judge_result,
            }
            
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write log: {e}")
    
    # Stop persistent Melvin process
    if persistent_graph:
        print("\nStopping persistent Melvin process...")
        stop_persistent_melvin()
        print("Done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Melvin Kindergarten Teacher")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    parser.add_argument("--tasks-per-round", type=int, default=3, help="Tasks per round")
    parser.add_argument("--melvin-binary", type=str, default="./melvin_learn_cli",
                       help="Path to melvin_learn_cli binary")
    parser.add_argument("--no-persistent", action="store_true",
                       help="Don't use persistent graph (fresh graph per task)")
    parser.add_argument("--graph-file", type=str, default=None,
                       help="Graph file path (default: melvin_global_graph.bin)")
    
    args = parser.parse_args()
    
    kindergarten_loop(
        num_rounds=args.rounds,
        tasks_per_round=args.tasks_per_round,
        melvin_binary=args.melvin_binary,
        persistent_graph=not args.no_persistent,
        graph_file=args.graph_file
    )

