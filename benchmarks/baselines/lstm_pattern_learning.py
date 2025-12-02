"""
Baseline Comparison: LSTM Pattern Learning

Train a simple LSTM to learn the same pattern "HELLO" and measure:
- Examples needed to reach 90% accuracy
- Training time
- Memory usage
- Model parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
from pathlib import Path

# Test pattern
PATTERN = "HELLO"

class SimpleLSTM(nn.Module):
    """Simple LSTM for sequence prediction"""
    def __init__(self, vocab_size, hidden_size=32):
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
    """Create training sequences from pattern"""
    # Convert to integer tokens
    chars = sorted(set(pattern + ' '))  # Include space as separator
    char_to_idx = {c: i for i, c in enumerate(chars)}
    
    sequences = []
    targets = []
    
    for _ in range(num_examples):
        # Input: H E L L -> Target: E L L O
        for i in range(len(pattern) - 1):
            sequences.append([char_to_idx[c] for c in pattern[:i+1]])
            targets.append(char_to_idx[pattern[i+1]])
    
    return sequences, targets, char_to_idx, chars

def train_and_test(num_examples):
    """Train LSTM on N examples and measure accuracy"""
    sequences, targets, char_to_idx, chars = create_training_data(PATTERN, num_examples)
    vocab_size = len(chars)
    
    # Create model
    model = SimpleLSTM(vocab_size, hidden_size=32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Training
    start_time = time.time()
    epochs = 50  # Fixed number of epochs
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for seq, target in zip(sequences, targets):
            # Pad sequence
            seq_tensor = torch.LongTensor([seq])
            target_tensor = torch.LongTensor([target])
            
            optimizer.zero_grad()
            output, _ = model(seq_tensor)
            loss = criterion(output[:, -1, :], target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    train_time = time.time() - start_time
    
    # Test accuracy: Can it predict the sequence?
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
    
    return {
        'examples': num_examples,
        'accuracy': accuracy,
        'train_time': train_time,
        'parameters': param_count,
        'memory_bytes': param_bytes,
        'epochs': epochs
    }

def main():
    print("==============================================")
    print("BASELINE: LSTM PATTERN LEARNING")
    print("==============================================\n")
    print(f"Pattern: \"{PATTERN}\"")
    print(f"Testing: Examples needed for 90% accuracy\n")
    
    # Create output directory
    Path("benchmarks/data").mkdir(parents=True, exist_ok=True)
    
    # CSV output
    csv_file = open("benchmarks/data/lstm_baseline_results.csv", "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["examples", "accuracy", "train_time_sec", "parameters", "memory_bytes", "epochs"])
    
    # Test with increasing examples
    for num_examples in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]:
        print(f"--- Test: {num_examples} examples ---")
        
        result = train_and_test(num_examples)
        
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  Train time: {result['train_time']:.3f}s")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Memory: {result['memory_bytes']:,} bytes\n")
        
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
    
    print("\n==============================================")
    print("Results saved to: benchmarks/data/lstm_baseline_results.csv")
    print("==============================================")

if __name__ == "__main__":
    main()

