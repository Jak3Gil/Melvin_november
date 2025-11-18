"""
Full-scale graph analysis script.
Analyzes the global graph state after training.
"""

import json
import sys
import subprocess
from collections import defaultdict
from typing import Dict, List, Any

def load_teacher_log(log_file: str = "teacher_log.jsonl") -> List[Dict]:
    """Load all entries from teacher log."""
    entries = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except FileNotFoundError:
        print(f"Log file {log_file} not found")
        return []
    return entries


def analyze_logs(entries: List[Dict]) -> Dict[str, Any]:
    """Analyze teacher log entries."""
    if not entries:
        return {}
    
    stats = {
        "total_rounds": len(set(e["round"] for e in entries)),
        "total_tasks": len(entries),
        "avg_score": 0.0,
        "score_distribution": defaultdict(int),
        "compression_stats": {"min": float('inf'), "max": 0.0, "avg": 0.0},
        "error_stats": {"min": float('inf'), "max": 0.0, "avg": 0.0},
        "pattern_quality_evolution": [],
        "task_types": defaultdict(int),
    }
    
    scores = []
    compressions = []
    errors = []
    
    for entry in entries:
        judge = entry.get("judge", {})
        melvin = entry.get("melvin_result", {})
        task = entry.get("task", {})
        
        score = judge.get("score", 0.0)
        scores.append(score)
        stats["score_distribution"][f"{int(score * 10) / 10:.1f}"] += 1
        
        comp = melvin.get("compression_ratio", 0.0)
        compressions.append(comp)
        
        err = melvin.get("reconstruction_error", 0.0)
        errors.append(err)
        
        patterns = melvin.get("patterns", [])
        if patterns:
            avg_q = sum(p.get("q", 0.0) for p in patterns) / len(patterns)
            stats["pattern_quality_evolution"].append({
                "round": entry["round"],
                "avg_q": avg_q,
                "num_patterns": len(patterns)
            })
        
        desc = task.get("description", "unknown")
        stats["task_types"][desc] += 1
    
    if scores:
        stats["avg_score"] = sum(scores) / len(scores)
    
    if compressions:
        stats["compression_stats"] = {
            "min": min(compressions),
            "max": max(compressions),
            "avg": sum(compressions) / len(compressions)
        }
    
    if errors:
        stats["error_stats"] = {
            "min": min(errors),
            "max": max(errors),
            "avg": sum(errors) / len(errors)
        }
    
    return stats


def analyze_graph_file(graph_file: str = "melvin_global_graph.bin", 
                      stats_binary: str = "../graph_stats") -> Dict[str, Any]:
    """Analyze the saved graph file by loading it and querying stats."""
    import os
    import subprocess
    
    if not os.path.exists(graph_file):
        return {"graph_file_exists": False}
    
    if not os.path.exists(stats_binary):
        return {
            "graph_file_exists": True,
            "note": "graph_stats binary not found - run 'make stats'"
        }
    
    try:
        result = subprocess.run(
            [stats_binary, graph_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        return {
            "graph_file_exists": True,
            "stats_output": result.stdout,
            "stats_error": result.stderr if result.stderr else None
        }
    except Exception as e:
        return {"error": str(e)}


def print_analysis(stats: Dict[str, Any], graph_info: Dict[str, Any]):
    """Print formatted analysis results."""
    print("=" * 60)
    print("MELVIN GLOBAL GRAPH ANALYSIS")
    print("=" * 60)
    print()
    
    print("--- Training Summary ---")
    print(f"Total rounds: {stats.get('total_rounds', 0)}")
    print(f"Total tasks: {stats.get('total_tasks', 0)}")
    print(f"Average judge score: {stats.get('avg_score', 0.0):.3f}")
    print()
    
    print("--- Score Distribution ---")
    dist = stats.get("score_distribution", {})
    for score_range in sorted(dist.keys(), key=float):
        count = dist[score_range]
        bar = "█" * (count // 2)
        print(f"  {score_range:>4}: {count:>3} {bar}")
    print()
    
    print("--- Compression Statistics ---")
    comp = stats.get("compression_stats", {})
    print(f"  Min: {comp.get('min', 0):.3f}")
    print(f"  Max: {comp.get('max', 0):.3f}")
    print(f"  Avg: {comp.get('avg', 0):.3f}")
    print()
    
    print("--- Reconstruction Error Statistics ---")
    err = stats.get("error_stats", {})
    print(f"  Min: {err.get('min', 0):.3f}")
    print(f"  Max: {err.get('max', 0):.3f}")
    print(f"  Avg: {err.get('avg', 0):.3f}")
    print()
    
    print("--- Pattern Quality Evolution ---")
    evolution = stats.get("pattern_quality_evolution", [])
    if evolution:
        # Show first, middle, last
        print(f"  Round {evolution[0]['round']}: avg_q={evolution[0]['avg_q']:.3f}, patterns={evolution[0]['num_patterns']}")
        if len(evolution) > 2:
            mid = len(evolution) // 2
            print(f"  Round {evolution[mid]['round']}: avg_q={evolution[mid]['avg_q']:.3f}, patterns={evolution[mid]['num_patterns']}")
        print(f"  Round {evolution[-1]['round']}: avg_q={evolution[-1]['avg_q']:.3f}, patterns={evolution[-1]['num_patterns']}")
    print()
    
    print("--- Task Type Distribution ---")
    task_types = stats.get("task_types", {})
    for desc, count in sorted(task_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {desc}: {count}")
    print()
    
    print("--- Graph File Info ---")
    if graph_info.get("graph_file_exists"):
        print("  Global graph file exists")
        if "stats_output" in graph_info:
            print("\n  Detailed Graph Statistics:")
            print(graph_info["stats_output"])
        elif "note" in graph_info:
            print(f"  {graph_info['note']}")
    else:
        print("  No global graph file found")
    print()
    
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Melvin training results")
    parser.add_argument("--log-file", type=str, default="teacher_log.jsonl",
                       help="Path to teacher log file")
    parser.add_argument("--graph-file", type=str, default="melvin_global_graph.bin",
                       help="Path to global graph file")
    
    args = parser.parse_args()
    
    # Load and analyze logs
    entries = load_teacher_log(args.log_file)
    stats = analyze_logs(entries)
    
    # Get graph info
    import os
    graph_info = {}
    if os.path.exists(args.graph_file):
        graph_info = analyze_graph_file(args.graph_file)
    else:
        graph_info = {"graph_file_exists": False}
    
    # Print analysis
    print_analysis(stats, graph_info)
    
    # Save detailed JSON report
    report = {
        "stats": stats,
        "graph_info": graph_info,
        "total_entries": len(entries)
    }
    
    with open("analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Detailed report saved to analysis_report.json")


if __name__ == "__main__":
    main()

