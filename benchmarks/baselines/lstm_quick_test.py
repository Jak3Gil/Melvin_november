"""
Quick LSTM test with progress bar - faster version for validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
from pathlib import Path

PATTERN = "HELLO"

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=16):  # Smaller for speed
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

def create_training_data(pattern, num_examples):
    chars = sorted(set(pattern + ' '))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    
    sequences = []
    targets = []
    
    for _ in range(num_examples):
        for i in range(len(pattern) - 1):
            sequences.append([char_to_idx[c] for c in pattern[:i+1]])
            targets.append(char_to_idx[pattern[i+1]])
    
    return sequences, targets, char_to_idx, chars

def train_and_test(num_examples):
    print(f"\n  [{num_examples:4d} examples] Training...", end='', flush=True)
    
    sequences, targets, char_to_idx, chars = create_training_data(PATTERN, num_examples)
    vocab_size = len(chars)
    
    model = SimpleLSTM(vocab_size, hidden_size=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    start_time = time.time()
    epochs = 20  # Reduced for speed
    
    for epoch in range(epochs):
        if epoch % 5 == 0:
            print('.', end='', flush=True)
        
        model.train()
        for seq, target in zip(sequences, targets):
            seq_tensor = torch.LongTensor([seq])
            target_tensor = torch.LongTensor([target])
            
            optimizer.zero_grad()
            output, _ = model(seq_tensor)
            loss = criterion(output[:, -1, :], target_tensor)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - start_time
    
    # Test
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(len(PATTERN) - 1):
            seq = [char_to_idx[c] for c in PATTERN[:i+1]]
            seq_tensor = torch.LongTensor([seq])
            output, _ = model(seq_tensor)
            pred = output[:, -1, :].argmax(dim=1).item()
            expected = char_to_idx[PATTERN[i+1]]
            
            if pred == expected:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f" Done! Acc={accuracy:.2%}, Time={train_time:.2f}s", flush=True)
    
    return {
        'examples': num_examples,
        'accuracy': accuracy,
        'train_time': train_time,
        'parameters': param_count,
        'memory_bytes': param_bytes,
        'epochs': epochs
    }

def main():
    print("=" * 60)
    print("QUICK LSTM BASELINE TEST (with progress)")
    print("=" * 60)
    print(f"Pattern: \"{PATTERN}\"")
    print(f"Testing: Examples needed for 90% accuracy")
    
    Path("benchmarks/data").mkdir(parents=True, exist_ok=True)
    
    csv_file = open("benchmarks/data/lstm_baseline_results.csv", "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["examples", "accuracy", "train_time_sec", "parameters", "memory_bytes", "epochs"])
    
    # Quick test with fewer points
    test_points = [1, 2, 3, 5, 10, 20, 50, 100]
    
    print(f"\nRunning {len(test_points)} tests...")
    
    for i, num_examples in enumerate(test_points, 1):
        print(f"[{i}/{len(test_points)}]", end='')
        result = train_and_test(num_examples)
        
        csv_writer.writerow([
            result['examples'],
            result['accuracy'],
            result['train_time'],
            result['parameters'],
            result['memory_bytes'],
            result['epochs']
        ])
        csv_file.flush()
    
    csv_file.close()
    
    print("\n" + "=" * 60)
    print("✓ Results saved to: benchmarks/data/lstm_baseline_results.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()

