#!/usr/bin/env python3
"""Quick script to show input/output examples"""

import json

entries = [json.loads(l) for l in open('teacher_log.jsonl') if l.strip()]
successful = [e for e in entries if 'error' not in e.get('melvin_result', {})]

print('=== Input/Output Examples ===\n')
for e in successful[:5]:
    print(f"Input: '{e['task']['input_str']}'")
    mr = e['melvin_result']
    print(f"  -> Created {mr.get('num_patterns', 0)} patterns")
    print(f"  -> Used {mr.get('explanation_apps', 0)} pattern applications")
    print(f"  -> Compression: {mr.get('compression_ratio', 0):.3f}")
    print(f"  -> Error: {mr.get('reconstruction_error', 0):.3f}")
    print()

